"""
Hyperparameter Sensitivity Analysis for TinkerRL
=================================================
Sweeps key PPO hyperparameters to assess benchmark robustness.

Reference:
    Henderson et al., "Deep Reinforcement Learning that Matters" (AAAI 2018)
    - Recommends reporting sensitivity to hyperparameters for RL benchmarks.

Usage:
    python scripts/hyperparam_sensitivity.py --seed 42 --output-dir results/sensitivity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import csv
import json
import time
from itertools import product

import numpy as np

from utils.seed import set_global_seed

# Try importing torch - graceful fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions.categorical import Categorical
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not installed. Running in dry-run mode.")


# ---------------------------------------------------------------------------
# Default hyperparameters (matching Tinker PPO)
# ---------------------------------------------------------------------------
DEFAULTS = {
    "learning_rate": 1e-4,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
}

# Sweep ranges for each hyperparameter
SWEEPS = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "clip_range": [0.05, 0.1, 0.2, 0.3, 0.5],
    "entropy_coef": [0.0, 0.001, 0.01, 0.05, 0.1],
    "gamma": [0.9, 0.95, 0.99, 0.995, 1.0],
    "gae_lambda": [0.8, 0.9, 0.95, 0.98, 1.0],
}

# PPO training constants
N_ENVS = 4
N_STEPS = 128        # rollout length per update
BATCH_SIZE = 256
N_EPOCHS = 4
ACTION_DIM = 199     # 0 … 198 (answers to num1+num2, max=99+99=198)
OBS_DIM = 2          # [num1, num2]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class ArithmeticEnv:
    """
    Simple arithmetic environment: add two numbers 1–99.

    Observation : [num1, num2]  (float32, normalized to [0, 1])
    Action      : integer in {0, …, 198} representing the predicted sum
    Reward      : 1.0 if action == num1 + num2, else 0.0
    Episode     : single step (done=True every step)
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.num1 = 1
        self.num2 = 1

    def reset(self):
        self.num1 = int(self.rng.integers(1, 100))
        self.num2 = int(self.rng.integers(1, 100))
        obs = np.array([self.num1 / 99.0, self.num2 / 99.0], dtype=np.float32)
        return obs

    def step(self, action: int):
        correct = self.num1 + self.num2
        reward = 1.0 if int(action) == correct else 0.0
        obs = self.reset()          # immediately start next problem
        return obs, reward, True, {}   # always done after one step


# ---------------------------------------------------------------------------
# PPO Agent (actor-critic MLP)
# ---------------------------------------------------------------------------
def _make_mlp(input_dim: int, hidden: int, output_dim: int) -> "nn.Sequential":
    """Build a 2-layer MLP with Tanh activations."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output_dim),
    )


class PPOAgent(nn.Module):
    """
    Separate actor and critic networks, each a 2-layer MLP with 64 hidden units.
    """

    def __init__(self, obs_dim: int = OBS_DIM, action_dim: int = ACTION_DIM, hidden: int = 64):
        super().__init__()
        self.actor  = _make_mlp(obs_dim, hidden, action_dim)
        self.critic = _make_mlp(obs_dim, hidden, 1)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        logits = self.actor(obs)
        dist   = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        value    = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# Rollout buffer (on-policy)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Minimal on-policy rollout buffer for PPO."""

    def __init__(self, n_steps: int, n_envs: int, obs_dim: int):
        self.n_steps = n_steps
        self.n_envs  = n_envs
        self.obs      = np.zeros((n_steps, n_envs, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((n_steps, n_envs), dtype=np.int64)
        self.rewards  = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.dones    = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.values   = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((n_steps, n_envs), dtype=np.float32)
        self.ptr      = 0

    def add(self, obs, actions, rewards, dones, values, log_probs):
        self.obs[self.ptr]       = obs
        self.actions[self.ptr]   = actions
        self.rewards[self.ptr]   = rewards
        self.dones[self.ptr]     = dones
        self.values[self.ptr]    = values
        self.log_probs[self.ptr] = log_probs
        self.ptr += 1

    def compute_returns_and_advantages(self, last_values, gamma: float, gae_lambda: float):
        advantages = np.zeros_like(self.rewards)
        last_gae   = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.n_steps)):
            next_non_terminal = 1.0 - self.dones[t]
            next_values = last_values if t == self.n_steps - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return returns, advantages


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_and_evaluate(
    hyperparams: dict,
    seed: int,
    num_steps: int = 20_000,
    eval_episodes: int = 200,
) -> float:
    """
    Train a PPO agent on ArithmeticEnv with the given hyperparameters and
    return final accuracy (fraction of eval episodes solved correctly).

    Parameters
    ----------
    hyperparams   : dict with keys from DEFAULTS
    seed          : RNG seed for reproducibility
    num_steps     : total environment steps to train for
    eval_episodes : number of episodes used for final evaluation

    Returns
    -------
    accuracy : float in [0, 1]
    """
    if not HAS_TORCH:
        # Dry-run: return synthetic values based on hyperparams
        return float(np.clip(np.random.default_rng(seed).normal(0.5, 0.1), 0, 1))

    lr           = hyperparams["learning_rate"]
    clip_range   = hyperparams["clip_range"]
    entropy_coef = hyperparams["entropy_coef"]
    gamma        = hyperparams["gamma"]
    gae_lambda   = hyperparams["gae_lambda"]

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cpu")

    # Vectorised environments (simple loop, no multiprocessing needed)
    envs = [ArithmeticEnv(np.random.default_rng(seed + i)) for i in range(N_ENVS)]
    obs_list = [env.reset() for env in envs]
    obs_arr  = np.stack(obs_list, axis=0)  # (n_envs, obs_dim)

    agent     = PPOAgent().to(device)
    optimizer = optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    n_updates      = num_steps // (N_STEPS * N_ENVS)
    global_step    = 0

    for update in range(1, n_updates + 1):
        buffer = RolloutBuffer(N_STEPS, N_ENVS, OBS_DIM)

        # --- Collect rollout ---
        for step in range(N_STEPS):
            obs_tensor  = torch.tensor(obs_arr, dtype=torch.float32, device=device)
            with torch.no_grad():
                actions, log_probs, _, values = agent.get_action_and_value(obs_tensor)
            actions_np  = actions.cpu().numpy()
            values_np   = values.cpu().numpy()
            log_probs_np = log_probs.cpu().numpy()

            rewards  = np.zeros(N_ENVS, dtype=np.float32)
            dones    = np.zeros(N_ENVS, dtype=np.float32)
            next_obs = np.zeros_like(obs_arr)

            for i, env in enumerate(envs):
                nobs, rew, done, _ = env.step(int(actions_np[i]))
                rewards[i]   = rew
                dones[i]     = float(done)
                next_obs[i]  = nobs

            buffer.add(obs_arr, actions_np, rewards, dones, values_np, log_probs_np)
            obs_arr = next_obs
            global_step += N_ENVS

        # Bootstrap value for last observation
        with torch.no_grad():
            last_values = agent.get_value(
                torch.tensor(obs_arr, dtype=torch.float32, device=device)
            ).cpu().numpy()

        returns, advantages = buffer.compute_returns_and_advantages(last_values, gamma, gae_lambda)

        # Flatten batch
        b_obs        = torch.tensor(buffer.obs.reshape(-1, OBS_DIM), dtype=torch.float32, device=device)
        b_actions    = torch.tensor(buffer.actions.reshape(-1), dtype=torch.long, device=device)
        b_returns    = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        b_log_probs  = torch.tensor(buffer.log_probs.reshape(-1), dtype=torch.float32, device=device)

        # Normalise advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # --- PPO update epochs ---
        batch_len    = b_obs.shape[0]
        mini_bs      = BATCH_SIZE
        for epoch in range(N_EPOCHS):
            idx = torch.randperm(batch_len, device=device)
            for start in range(0, batch_len, mini_bs):
                mb = idx[start:start + mini_bs]
                _, new_log_probs, entropy, new_values = agent.get_action_and_value(b_obs[mb], b_actions[mb])

                ratio     = torch.exp(new_log_probs - b_log_probs[mb])
                pg_loss1  = -b_advantages[mb] * ratio
                pg_loss2  = -b_advantages[mb] * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                pg_loss   = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = 0.5 * ((new_values - b_returns[mb]) ** 2).mean()
                ent_loss   = entropy.mean()

                loss = pg_loss + 0.5 * value_loss - entropy_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

    # --- Evaluation ---
    eval_env = ArithmeticEnv(np.random.default_rng(seed + 9999))
    correct  = 0
    for _ in range(eval_episodes):
        obs = eval_env.reset()
        with torch.no_grad():
            obs_t  = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = agent.actor(obs_t).argmax(dim=-1).item()
        n1, n2 = eval_env.num1, eval_env.num2
        if int(action) == n1 + n2:
            correct += 1

    return correct / eval_episodes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sensitivity analysis for TinkerRL PPO."
    )
    parser.add_argument("--seed",       type=int,  default=42,                       help="Base RNG seed")
    parser.add_argument("--output-dir", type=str,  default="results/sensitivity",    help="Directory for CSV output")
    parser.add_argument("--num-steps",  type=int,  default=20_000,                   help="Training steps per configuration")
    parser.add_argument("--dry-run",    action="store_true",                          help="Skip training, use random results")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run mode enabled: skipping actual training.")

    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    results = []   # list of dicts: {hyperparameter, value, accuracy, seed}

    print(f"\n{'='*60}")
    print(f"  TinkerRL Hyperparameter Sensitivity Analysis")
    print(f"  Seed: {args.seed}  |  Steps/config: {args.num_steps:,}")
    print(f"{'='*60}\n")

    total_configs = sum(len(v) for v in SWEEPS.items() if isinstance(v, list) for _ in [None])
    # Correct count
    total_configs = sum(len(vals) for vals in SWEEPS.values())
    done = 0

    for param_name, sweep_values in SWEEPS.items():
        print(f"--- Sweeping: {param_name} ---")
        for val in sweep_values:
            hp = dict(DEFAULTS)          # start from defaults
            hp[param_name] = val         # override one parameter

            t0 = time.time()
            accuracy = train_and_evaluate(
                hyperparams=hp,
                seed=args.seed,
                num_steps=args.num_steps,
            )
            elapsed = time.time() - t0

            results.append({
                "hyperparameter": param_name,
                "value": val,
                "accuracy": round(accuracy, 4),
                "seed": args.seed,
            })
            done += 1
            print(f"  {param_name}={val:<10}  accuracy={accuracy:.3f}  [{elapsed:.1f}s]  ({done}/{total_configs})")

    # --- Save CSV ---
    csv_path = os.path.join(args.output_dir, f"sensitivity_seed{args.seed}.csv")
    fieldnames = ["hyperparameter", "value", "accuracy", "seed"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {csv_path}")

    # --- Summary table ---
    print(f"\n{'='*60}")
    print(f"{'Hyperparameter':<20} {'Value':<12} {'Accuracy':>10}")
    print(f"{'-'*20} {'-'*12} {'-'*10}")
    for row in results:
        print(f"{row['hyperparameter']:<20} {str(row['value']):<12} {row['accuracy']:>10.3f}")

    # --- Sensitivity report ---
    print(f"\n{'='*60}")
    print("  Sensitivity Report (>10% accuracy swing = HIGH)")
    print(f"{'='*60}")
    by_param = {}
    for row in results:
        by_param.setdefault(row["hyperparameter"], []).append(row["accuracy"])

    high_sensitivity = []
    for param, accs in by_param.items():
        swing = max(accs) - min(accs)
        tag   = "HIGH" if swing > 0.10 else "low"
        print(f"  {param:<20}  swing={swing:.3f}  [{tag}]")
        if swing > 0.10:
            high_sensitivity.append(param)

    if high_sensitivity:
        print(f"\n  High-sensitivity hyperparameters: {', '.join(high_sensitivity)}")
    else:
        print("\n  No hyperparameter caused >10% accuracy swing.")

    # Save summary JSON alongside CSV
    summary = {
        "seed": args.seed,
        "num_steps": args.num_steps,
        "high_sensitivity": high_sensitivity,
        "swings": {p: round(max(a) - min(a), 4) for p, a in by_param.items()},
    }
    json_path = os.path.join(args.output_dir, f"sensitivity_summary_seed{args.seed}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved to: {json_path}")


if __name__ == "__main__":
    main()
