"""100-step GRPO on xlam-60k real data — optimized for speed"""

import os, json, re, warnings, random

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer
from datasets import load_dataset

EXP = "xlam100"
MODEL = "Qwen/Qwen3-8B"
STEPS, GROUP, LR, TEMP = 100, 4, 3e-5, 0.8
SAVE_EVERY = 25

SYSTEM_PROMPT = 'You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n{"tool": "<name>", "arguments": {<key>: <value>}}\nNo prose. Only JSON.'

print(f"[{EXP}] Loading xlam-60k...")
ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
examples = []
for row in ds:
    try:
        query = row.get("query", row.get("instruction", ""))
        tools = (
            json.loads(row.get("tools", "[]"))
            if isinstance(row.get("tools"), str)
            else row.get("tools", [])
        )
        answers = (
            json.loads(row.get("answers", "[]"))
            if isinstance(row.get("answers"), str)
            else row.get("answers", [])
        )
        if not answers or not isinstance(answers, list):
            continue
        ans = answers[0]
        tool_name = ans.get("name", ans.get("tool", ""))
        arguments = ans.get("arguments", ans.get("parameters", {}))
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not tool_name:
            continue
        tool_schema = json.dumps(tools[:8])
        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAvailable tools:\n{tool_schema}\n\nUser: {query}<|im_end|>\n<|im_start|>assistant\n"
        examples.append((prompt, tool_name, arguments))
    except:
        continue
random.shuffle(examples)
examples = examples[:3000]
print(f"[{EXP}] {len(examples)} examples")


def reward(response, tool_name, arguments):
    m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except:
        return 0.1
    score = 0.3
    if p.get("tool") == tool_name or p.get("name") == tool_name:
        score += 0.4
    pred_args = p.get("arguments", p.get("parameters", {}))
    if isinstance(pred_args, dict) and arguments:
        score += 0.3 * sum(1 for k in arguments if k in pred_args) / len(arguments)
    return min(score, 1.0)


_adv = []


def loss_fn(data, lp):
    losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
    loss = torch.stack(losses).mean()
    return loss, {"loss": loss.item()}


svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=32)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
w0 = tc.save_weights_for_sampler(name="step_0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"[{EXP}] Run: {tc.model_id}")

step_rewards = []
for step in range(STEPS):
    batch = random.sample(examples, 2)  # 2 prompts for speed
    all_data, all_advs, batch_r = [], [], []
    for prompt_text, tn, args in batch:
        pid = tok.encode(prompt_text, add_special_tokens=False)
        if len(pid) > 1536:
            pid = pid[:1536]
        sp = T.SamplingParams(max_tokens=128, temperature=TEMP, top_p=0.95)
        resp = sc.sample(
            T.ModelInput.from_ints(pid), num_samples=GROUP, sampling_params=sp
        ).result()
        rews = [
            reward(tok.decode(list(r.tokens), skip_special_tokens=True), tn, args)
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
    print(
        f"[{EXP}] {step + 1:3d}/{STEPS} | loss={result.metrics.get('loss', 0):.4f} | reward={avg:.3f}"
    )
    if (step + 1) % SAVE_EVERY == 0:
        tc.save_state(name=f"s{step + 1}")
        ckpt = tc.save_weights_for_sampler(name=f"s{step + 1}").result()
        sc = tc.create_sampling_client(model_path=ckpt.path)
        print(f"[{EXP}]   -> ckpt s{step + 1}")

tc.save_state(name="final")
f = tc.save_weights_for_sampler(name="final").result()
last10 = step_rewards[-10:]
avg10 = sum(last10) / len(last10) if last10 else 0
print(f"\n[{EXP}] DONE | last10={avg10:.3f} | run={tc.model_id} | path={f.path}")
