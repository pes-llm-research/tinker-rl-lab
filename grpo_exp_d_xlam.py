"""Experiment D: GRPO with Salesforce/xlam-function-calling-60k real dataset"""

import os, json, re, warnings, random

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)

import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

EXP_NAME = "D_xlam"
MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32
GROUP_SIZE = 8
STEPS = 30
LR = 3e-5
TEMP = 0.8
SAVE_EVERY = 10

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
    '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
    "No prose. Only JSON."
)

# ── Load real dataset ────────────────────────────────────────────────────
print(f"[{EXP_NAME}] Loading Salesforce/xlam-function-calling-60k...")
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
print(f"[{EXP_NAME}] Loaded {len(ds)} examples")

# Parse dataset into (prompt, tool_name, arguments) triples
examples = []
for row in ds:
    try:
        query = row.get("query", row.get("instruction", ""))
        tools_str = row.get("tools", "[]")
        answers_str = row.get("answers", "[]")
        if not query or not answers_str:
            continue
        tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str
        if not answers or not isinstance(answers, list):
            continue
        # Take first answer
        ans = answers[0]
        tool_name = ans.get("name", ans.get("tool", ""))
        arguments = ans.get("arguments", ans.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not tool_name:
            continue
        # Build prompt with available tools
        tool_schema = json.dumps(tools[:10]) if tools else "[]"  # limit to 10 tools
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\nAvailable tools:\n{tool_schema}\n\nUser: {query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        examples.append((prompt, tool_name, arguments))
    except (json.JSONDecodeError, KeyError, TypeError):
        continue

random.shuffle(examples)
examples = examples[:2000]  # cap at 2000 for speed
print(f"[{EXP_NAME}] Parsed {len(examples)} usable examples")


# ── Reward function ──────────────────────────────────────────────────────
def reward(response, tool_name, arguments):
    m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except json.JSONDecodeError:
        return 0.1
    score = 0.3
    if p.get("tool") == tool_name or p.get("name") == tool_name:
        score += 0.4
    pred_args = p.get("arguments", p.get("parameters", {}))
    if isinstance(pred_args, dict) and arguments:
        matched = sum(1 for k in arguments if k in pred_args)
        score += 0.3 * matched / len(arguments)
    return min(score, 1.0)


_advantages = []


def grpo_loss_fn(data, logprobs_list):
    losses = []
    for i, logprobs in enumerate(logprobs_list):
        losses.append(-_advantages[i] * logprobs.sum())
    loss = torch.stack(losses).mean()
    return loss, {"grpo_loss": loss.item()}


# ── Setup ────────────────────────────────────────────────────────────────
print(f"[{EXP_NAME}] Connecting...")
svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
w0 = tc.save_weights_for_sampler(name="step_0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"[{EXP_NAME}] Run: {tc.model_id} | LR={LR} group={GROUP_SIZE} temp={TEMP}")

# ── GRPO loop ────────────────────────────────────────────────────────────
step_rewards = []
for step in range(STEPS):
    batch = random.sample(examples, 4)
    all_data, all_advs, batch_rewards = [], [], []

    for prompt_text, tool_name, args in batch:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        # Truncate if too long (xlam can have long tool schemas)
        if len(prompt_ids) > 2048:
            prompt_ids = prompt_ids[:2048]
        sp = T.SamplingParams(max_tokens=256, temperature=TEMP, top_p=0.95)
        responses = sc.sample(
            T.ModelInput.from_ints(prompt_ids), num_samples=GROUP_SIZE, sampling_params=sp
        ).result()
        rewards = []
        for resp in responses.sequences:
            text = tok.decode(list(resp.tokens), skip_special_tokens=True)
            rewards.append(reward(text, tool_name, args))
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
        advs = [(r - mean_r) / std_r for r in rewards]
        batch_rewards.extend(rewards)
        for resp, adv in zip(responses.sequences, advs):
            resp_ids = list(resp.tokens)
            full_ids = prompt_ids + resp_ids
            target_ids = full_ids[1:] + [0]
            all_data.append(
                T.Datum(
                    model_input=T.ModelInput.from_ints(full_ids),
                    loss_fn_inputs={
                        "target_tokens": T.TensorData(
                            data=target_ids, dtype="int64", shape=[len(target_ids)]
                        )
                    },
                )
            )
            all_advs.append(adv)

    if not all_data:
        continue
    _advantages = all_advs
    result = tc.forward_backward_custom(data=all_data, loss_fn=grpo_loss_fn).result()
    tc.optim_step(T.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)).result()
    avg_r = sum(batch_rewards) / len(batch_rewards)
    step_rewards.append(avg_r)
    loss_val = result.metrics.get("grpo_loss", float("nan"))
    print(f"[{EXP_NAME}] Step {step + 1:3d}/{STEPS} | loss={loss_val:.4f} | reward={avg_r:.3f}")
    if (step + 1) % SAVE_EVERY == 0:
        tc.save_state(name=f"state_{step + 1}")
        ckpt = tc.save_weights_for_sampler(name=f"step_{step + 1}").result()
        sc = tc.create_sampling_client(model_path=ckpt.path)
        print(f"[{EXP_NAME}]   -> Checkpoint step_{step + 1}")

tc.save_state(name="final")
final = tc.save_weights_for_sampler(name="final").result()
print(
    f"\n[{EXP_NAME}] DONE | avg_reward_last10={sum(step_rewards[-10:]) / max(len(step_rewards[-10:]), 1):.3f}"
)
print(f"[{EXP_NAME}] Run ID: {tc.model_id}")
print(f"[{EXP_NAME}] Sampler: {final.path}")
