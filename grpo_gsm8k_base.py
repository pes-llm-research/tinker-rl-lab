"""GRPO on GSM8K — parameterized for multi-seed, scaling, and ablation runs.
Usage: python grpo_gsm8k_base.py --model Qwen/Qwen3-8B --seed 137 --rank 32 --steps 50
"""

import os, json, re, warnings, random, argparse

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="Qwen/Qwen3-8B")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--lr", type=float, default=3e-5)
parser.add_argument("--group", type=int, default=4)
parser.add_argument("--batch", type=int, default=2)
parser.add_argument("--tag", default="")
args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

MODEL = args.model
EXP = args.tag or f"gsm8k_{MODEL.split('/')[-1]}_s{args.seed}_r{args.rank}"
STEPS, GROUP, LR, RANK = args.steps, args.group, args.lr, args.rank
SAVE_EVERY = max(args.steps // 4, 10)

SYSTEM_PROMPT = "You are a math assistant. Solve the problem step by step, then give your final numerical answer inside \\boxed{}."

# ── Load GSM8K ───────────────────────────────────────────────────────────
print(f"[{EXP}] Loading GSM8K...")
ds = load_dataset("openai/gsm8k", "main", split="train")
examples = []
for row in ds:
    q = row["question"]
    # Extract final numeric answer from "#### <number>"
    ans_match = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
    if not ans_match:
        continue
    answer = ans_match.group(1).replace(",", "").strip()
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    examples.append((prompt, answer))

random.shuffle(examples)
print(
    f"[{EXP}] {len(examples)} GSM8K examples | model={MODEL} seed={args.seed} rank={RANK} lr={LR}"
)


# ── Reward: binary exact match on \\boxed{} or final number ─────────────
def reward(response, answer):
    response = response.strip()
    # Check \boxed{answer}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except:
            if b_clean == answer:
                return 1.0
    # Check last number in response
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except:
            pass
    return 0.0  # Binary reward


# ── GRPO ─────────────────────────────────────────────────────────────────
_adv = []


def loss_fn(data, lp):
    losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
    loss = torch.stack(losses).mean()
    return loss, {"loss": loss.item()}


print(f"[{EXP}] Connecting to Tinker...")
svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=RANK)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
w0 = tc.save_weights_for_sampler(name="s0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"[{EXP}] Run: {tc.model_id}")

# ── Training loop ────────────────────────────────────────────────────────
step_rewards = []
zero_loss_steps = 0
zero_reward_steps = 0

for step in range(STEPS):
    batch = random.sample(examples, args.batch)
    all_data, all_advs, batch_r = [], [], []

    for prompt_text, ans in batch:
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > 1024:
            pid = pid[:1024]
        sp = T.SamplingParams(max_tokens=512, temperature=0.8, top_p=0.95)
        resp = sc.sample(
            T.ModelInput.from_ints(pid), num_samples=GROUP, sampling_params=sp
        ).result()
        rews = [
            reward(tok.decode(list(r.tokens), skip_special_tokens=True), ans)
            for r in resp.sequences
        ]
        mr = sum(rews) / len(rews)
        sr = (sum((r - mr) ** 2 for r in rews) / len(rews)) ** 0.5 + 1e-8
        advs = [(r - mr) / sr for r in rews]
        batch_r.extend(rews)
        for r, a in zip(resp.sequences, advs):
            rid = list(r.tokens)
            fid = pid + rid
            tid = fid[1:] + [0]
            all_data.append(
                T.Datum(
                    model_input=T.ModelInput.from_ints(fid),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(data=tid, dtype="int64", shape=[len(tid)])
                    },
                )
            )
            all_advs.append(a)

    if not all_data:
        continue

    _adv = all_advs
    result = tc.forward_backward_custom(data=all_data, loss_fn=loss_fn).result()
    tc.optim_step(T.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)).result()

    avg = sum(batch_r) / len(batch_r)
    step_rewards.append(avg)
    loss_val = result.metrics.get("loss", 0)
    if abs(loss_val) < 1e-6:
        zero_loss_steps += 1
    if avg == 0:
        zero_reward_steps += 1

    print(
        f"[{EXP}] {step + 1:3d}/{STEPS} | loss={loss_val:.4f} | reward={avg:.3f} | acc={avg * 100:.1f}%"
    )

    if (step + 1) % SAVE_EVERY == 0:
        tc.save_state(name=f"s{step + 1}")
        ckpt = tc.save_weights_for_sampler(name=f"s{step + 1}").result()
        sc = tc.create_sampling_client(model_path=ckpt.path)
        print(f"[{EXP}]   -> ckpt s{step + 1}")

# ── Final stats ──────────────────────────────────────────────────────────
tc.save_state(name="final")
f = tc.save_weights_for_sampler(name="final").result()
last10 = step_rewards[-10:]
avg10 = sum(last10) / len(last10) if last10 else 0
first5 = step_rewards[:5]
avg_first5 = sum(first5) / len(first5) if first5 else 0
max_r = max(step_rewards) if step_rewards else 0

print(f"\n[{EXP}] === FINAL REPORT ===")
print(f"[{EXP}] Model: {MODEL} | Seed: {args.seed} | LoRA rank: {RANK}")
print(f"[{EXP}] Steps: {STEPS} | Group: {GROUP} | LR: {LR}")
print(f"[{EXP}] First-5 avg accuracy: {avg_first5 * 100:.1f}%")
print(f"[{EXP}] Last-10 avg accuracy: {avg10 * 100:.1f}%")
print(f"[{EXP}] Peak accuracy: {max_r * 100:.1f}%")
print(f"[{EXP}] Zero-loss steps: {zero_loss_steps}/{STEPS} ({100 * zero_loss_steps / STEPS:.0f}%)")
print(
    f"[{EXP}] Zero-reward steps: {zero_reward_steps}/{STEPS} ({100 * zero_reward_steps / STEPS:.0f}%)"
)
print(f"[{EXP}] Run ID: {tc.model_id}")
print(f"[{EXP}] Sampler: {f.path}")
print(f"[{EXP}] Reward trace: {[round(r, 3) for r in step_rewards]}")
