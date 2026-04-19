"""
GRPO tool-use fine-tuning via Tinker SDK.
Model: Qwen/Qwen3-8B, LoRA rank 32
Task:  single-turn structured tool calling (5 tools)
Loss:  forward_backward_custom with GRPO advantage normalization
"""

import os, json, re, warnings, random, time

warnings.filterwarnings("ignore")
if "TINKER_API_KEY" not in os.environ:
    raise RuntimeError("Set TINKER_API_KEY env var (already in ~/.zshrc)")

import torch
import tinker
import tinker.types as T
from transformers import AutoTokenizer

# ── Config ───────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-8B"
LORA_RANK = 32
GROUP_SIZE = 8
STEPS = 50
LR = 3e-5
SAVE_EVERY = 10

SYSTEM_PROMPT = (
    "You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n"
    '{"tool": "<name>", "arguments": {<key>: <value>}}\n'
    "No prose. Only JSON."
)

# ── Synthetic dataset ─────────────────────────────────────────────────────
TOOLS = [
    {"name": "calculator", "description": "Arithmetic", "parameters": {"expression": "string"}},
    {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"city": "string", "units": "string"},
    },
    {"name": "web_search", "description": "Web search", "parameters": {"query": "string"}},
    {"name": "get_time", "description": "Time in timezone", "parameters": {"timezone": "string"}},
    {
        "name": "set_reminder",
        "description": "Set a reminder",
        "parameters": {"task": "string", "time": "string"},
    },
]
TOOL_SCHEMA = json.dumps(TOOLS)

RAW = [
    ("What is 245 * 37?", "calculator", {"expression": "245 * 37"}),
    ("Calculate sqrt(144)", "calculator", {"expression": "sqrt(144)"}),
    ("15% of 980?", "calculator", {"expression": "0.15 * 980"}),
    ("Divide 1024 by 32", "calculator", {"expression": "1024 / 32"}),
    ("2 to the power of 10", "calculator", {"expression": "2 ** 10"}),
    ("Weather in Tokyo?", "get_weather", {"city": "Tokyo", "units": "metric"}),
    ("Is it raining in London?", "get_weather", {"city": "London", "units": "metric"}),
    ("Temperature in New York", "get_weather", {"city": "New York", "units": "imperial"}),
    ("How hot is Dubai right now?", "get_weather", {"city": "Dubai", "units": "metric"}),
    ("Search for GPT-5 news", "web_search", {"query": "GPT-5 news"}),
    ("Capital of Australia?", "web_search", {"query": "capital of Australia"}),
    ("Find Python asyncio tutorial", "web_search", {"query": "Python asyncio tutorial"}),
    ("What time is it in Singapore?", "get_time", {"timezone": "Asia/Singapore"}),
    ("Current time in Los Angeles?", "get_time", {"timezone": "America/Los_Angeles"}),
    ("Time in Berlin?", "get_time", {"timezone": "Europe/Berlin"}),
    ("Remind me to call mom at 6pm", "set_reminder", {"task": "call mom", "time": "6pm"}),
    (
        "Set a reminder for team meeting 10am",
        "set_reminder",
        {"task": "team meeting", "time": "10am"},
    ),
    ("Remind me to take medicine at 8pm", "set_reminder", {"task": "take medicine", "time": "8pm"}),
]


def make_prompt(query):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nAvailable tools:\n{TOOL_SCHEMA}\n\nUser: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


examples = [(make_prompt(q), t, a) for q, t, a in RAW] * 28
random.shuffle(examples)
print(f"Dataset: {len(examples)} examples, {len(set(t for _, t, _ in RAW))} tools")


# ── Reward function ───────────────────────────────────────────────────────
def reward(response: str, tool_name: str, arguments: dict) -> float:
    m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except json.JSONDecodeError:
        return 0.1
    score = 0.3  # valid JSON
    if p.get("tool") == tool_name or p.get("name") == tool_name:
        score += 0.4
    pred_args = p.get("arguments", p.get("parameters", {}))
    if isinstance(pred_args, dict) and arguments:
        score += 0.3 * sum(1 for k in arguments if k in pred_args) / len(arguments)
    return min(score, 1.0)


# ── GRPO loss using forward_backward_custom ───────────────────────────────
# Store advantages in a side-channel since Tinker only allows
# target_tokens + weights in loss_fn_inputs.
_advantages = []


def grpo_loss_fn(data, logprobs_list):
    """Compute GRPO policy gradient loss from per-token logprobs."""
    losses = []
    for i, logprobs in enumerate(logprobs_list):
        adv = _advantages[i]
        # sum log probs over response tokens, scale by advantage
        losses.append(-adv * logprobs.sum())
    loss = torch.stack(losses).mean()
    return loss, {"grpo_loss": loss.item()}


# ── Tinker setup ─────────────────────────────────────────────────────────
print("Connecting to Tinker...")
svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=LORA_RANK)
print(f"Run ID: {tc.model_id}")

print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

print("Saving initial sampler weights...")
w0 = tc.save_weights_for_sampler(name="step_0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"Ready. Sampler: {w0.path}")

# ── GRPO loop ─────────────────────────────────────────────────────────────
print(f"\nGRPO — {STEPS} steps, group={GROUP_SIZE}, model={MODEL}\n")

step_rewards = []

for step in range(STEPS):
    batch = random.sample(examples, 4)  # 4 prompts per step

    all_data = []
    all_advs = []
    batch_rewards = []

    for prompt_text, tool_name, args in batch:
        prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
        prompt_mi = T.ModelInput.from_ints(prompt_ids)

        # Sample GROUP_SIZE completions
        sp = T.SamplingParams(max_tokens=192, temperature=0.8, top_p=0.95)
        responses = sc.sample(prompt_mi, num_samples=GROUP_SIZE, sampling_params=sp).result()

        rewards = []
        for resp in responses.sequences:
            resp_ids = list(resp.tokens)
            text = tok.decode(resp_ids, skip_special_tokens=True)
            r = reward(text, tool_name, args)
            rewards.append(r)

        # Group-relative advantages
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-8
        advs = [(r - mean_r) / std_r for r in rewards]
        batch_rewards.extend(rewards)

        for resp, adv in zip(responses.sequences, advs):
            resp_ids = list(resp.tokens)
            full_ids = prompt_ids + resp_ids
            # target_tokens: shifted input for next-token prediction
            # For causal LM: target[i] = input[i+1]
            target_ids = full_ids[1:] + [0]  # pad last position
            datum = T.Datum(
                model_input=T.ModelInput.from_ints(full_ids),
                loss_fn_inputs={
                    "target_tokens": T.TensorData(
                        data=target_ids, dtype="int64", shape=[len(target_ids)]
                    ),
                },
            )
            all_data.append(datum)
            all_advs.append(adv)

    if not all_data:
        continue

    # Set advantages for the loss function
    _advantages = all_advs

    # Forward-backward with GRPO loss
    result = tc.forward_backward_custom(
        data=all_data,
        loss_fn=grpo_loss_fn,
        loss_type_input="logprobs",
    ).result()
    tc.optim_step(T.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95, eps=1e-8)).result()

    avg_r = sum(batch_rewards) / len(batch_rewards)
    step_rewards.append(avg_r)
    grpo_loss_val = result.metrics.get("grpo_loss", float("nan"))
    print(f"Step {step + 1:3d}/{STEPS} | loss={grpo_loss_val:.4f} | reward={avg_r:.3f}")

    if (step + 1) % SAVE_EVERY == 0:
        state = tc.save_state(name=f"state_{step + 1}")
        ckpt = tc.save_weights_for_sampler(name=f"step_{step + 1}").result()
        sc = tc.create_sampling_client(model_path=ckpt.path)
        print(f"  → Saved: {state} | Sampler: step_{step + 1}")

# ── Final save ────────────────────────────────────────────────────────────
final_state = tc.save_state(name="final")
final_ckpt = tc.save_weights_for_sampler(name="final").result()
print(f"\nFinal checkpoint: {final_state}")
print(f"Sampler weights:  {final_ckpt.path}")
print(f"Avg reward last 10: {sum(step_rewards[-10:]) / max(len(step_rewards[-10:]), 1):.3f}")
print(f"Run ID: {tc.model_id}")
print("Done.")
