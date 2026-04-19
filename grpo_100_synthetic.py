"""100-step GRPO on synthetic 5-tool data — optimized for speed"""

import os, json, re, warnings, random

warnings.filterwarnings("ignore")
assert os.environ.get("TINKER_API_KEY"), (
    "Set TINKER_API_KEY in env (was hardcoded, removed 2026-04-11)"
)
import torch, tinker, tinker.types as T
from transformers import AutoTokenizer

EXP = "synth100"
MODEL = "Qwen/Qwen3-8B"
STEPS, GROUP, LR, TEMP = 100, 4, 3e-5, 0.8
SAVE_EVERY = 25

SYSTEM_PROMPT = 'You are a tool-calling assistant. Respond ONLY with a valid JSON object:\n{"tool": "<name>", "arguments": {<key>: <value>}}\nNo prose. Only JSON.'
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
TS = json.dumps(TOOLS)
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


def mkp(q):
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nAvailable tools:\n{TS}\n\nUser: {q}<|im_end|>\n<|im_start|>assistant\n"


examples = [(mkp(q), t, a) for q, t, a in RAW] * 28
random.shuffle(examples)


def reward(response, tn, args):
    m = re.search(r"\{.*\}", response.strip(), re.DOTALL)
    if not m:
        return 0.0
    try:
        p = json.loads(m.group())
    except:
        return 0.1
    s = 0.3
    if p.get("tool") == tn or p.get("name") == tn:
        s += 0.4
    pa = p.get("arguments", p.get("parameters", {}))
    if isinstance(pa, dict) and args:
        s += 0.3 * sum(1 for k in args if k in pa) / len(args)
    return min(s, 1.0)


_adv = []


def loss_fn(data, lp):
    losses = [(-_adv[i] * lp[i].sum()) for i in range(len(lp))]
    loss = torch.stack(losses).mean()
    return loss, {"loss": loss.item()}


svc = tinker.ServiceClient(base_url=None)
tc = svc.create_lora_training_client(base_model=MODEL, rank=32)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
w0 = tc.save_weights_for_sampler(name="s0").result()
sc = tc.create_sampling_client(model_path=w0.path)
print(f"[{EXP}] Run: {tc.model_id}")

step_rewards = []
for step in range(STEPS):
    batch = random.sample(examples, 2)
    all_data, all_advs, batch_r = [], [], []
    for pt, tn, args in batch:
        pid = tok.encode(pt, add_special_tokens=False)
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
