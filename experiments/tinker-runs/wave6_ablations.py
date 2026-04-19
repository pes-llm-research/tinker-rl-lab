#!/usr/bin/env python3
"""
WAVE 6 — Qwen3-8B GSM8K sensitivity ablations.

Three 1-D sweeps around a shared baseline (rank=32, temperature=0.8, batch=2)
using the same GRPO training loop as campaign_v2.py:

  • Temperature sweep  — {0.2, 0.4, 0.6, 0.8, 1.0} (at rank=32, batch=2)
  • LoRA rank sweep    — {4, 8, 16, 32, 64}        (at temperature=0.8, batch=2)
  • Batch size sweep   — {1, 2, 4, 8}               (at rank=32, temperature=0.8)

All runs: model=Qwen/Qwen3-8B, task=GSM8K, seed=42, steps=30, group=8, lr=1e-5.
Shared baseline is run once and re-used across all three sweeps.

Output: experiments/tinker-runs/results/wave6_ablations.json
"""

import json
import os
import random
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import torch
import tinker
import tinker.types as T
from datasets import load_dataset
from transformers import AutoTokenizer

warnings.filterwarnings("ignore")

API_KEY = os.environ.get("TINKER_API_KEY", "")
WANDB_KEY = os.environ.get("WANDB_API_KEY", "")
os.environ["TINKER_API_KEY"] = API_KEY


# ── Constants ───────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-8B"
MODEL_SHORT = "qwen3-8b"
SEED = 42
STEPS = 30
GROUP_SIZE = 8
LR = 1e-5

# Baseline values (shared across sweeps)
BASE_RANK = 32
BASE_TEMP = 0.8
BASE_BATCH = 2

# Default output path. Derived from this file's location so the script
# is portable across checkouts; overridable via --out when running the
# figure generator, or by setting WAVE6_RESULTS_PATH in the environment.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESULTS_PATH = os.environ.get(
    "WAVE6_RESULTS_PATH",
    os.path.join(
        _REPO_ROOT,
        "experiments",
        "tinker-runs",
        "results",
        "wave6_ablations.json",
    ),
)

SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, "
    "then give your final numerical answer inside \\boxed{}."
)
QUESTION_SUFFIX = (
    " Provide a numerical answer without units, written inside \\boxed{}."
)


# ── Lazy GSM8K loader ───────────────────────────────────────────────
# Dataset loading is deferred until a worker actually needs it so that
# `import wave6_ablations` stays cheap (no network/disk at import time).
_GSM8K_CACHE = []


def _load_gsm8k():
    if _GSM8K_CACHE:
        return _GSM8K_CACHE
    print("Loading GSM8K dataset...", flush=True)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if m:
            _GSM8K_CACHE.append(
                (row["question"], m.group(1).replace(",", "").strip())
            )
    print(f"Loaded {len(_GSM8K_CACHE)} GSM8K examples", flush=True)
    return _GSM8K_CACHE


def reward_fn(response: str, answer: str) -> float:
    response = response.strip()
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    for b in boxed:
        b_clean = b.strip().replace(",", "").replace(" ", "")
        try:
            if abs(float(b_clean) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            if b_clean == answer:
                return 1.0
    all_nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", response)
    if all_nums:
        last = all_nums[-1].replace(",", "")
        try:
            if abs(float(last) - float(answer)) < 0.01:
                return 1.0
        except Exception:
            pass
    return 0.0


# ── One GRPO run ────────────────────────────────────────────────────
def run_one(exp):
    tag = exp["tag"]
    rank = exp["rank"]
    temperature = exp["temperature"]
    batch = exp["batch"]
    seed = exp.get("seed", SEED)
    steps = exp.get("steps", STEPS)
    group_size = exp.get("group_size", GROUP_SIZE)
    lr = exp.get("lr", LR)

    print(
        f"  >> [{tag}] START rank={rank} temp={temperature} batch={batch} "
        f"G={group_size} steps={steps}",
        flush=True,
    )

    wb_run = None
    t0 = time.time()
    try:
        random.seed(seed)
        torch.manual_seed(seed)

        gsm8k = _load_gsm8k()

        tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

        # W&B
        try:
            import wandb

            if WANDB_KEY:
                wandb.login(key=WANDB_KEY, relogin=True)
            wb_run = wandb.init(
                project="tinker-rl-lab-world-class",
                name=f"wave6_{tag}",
                config={
                    "model": MODEL,
                    "method": "grpo",
                    "task": "gsm8k",
                    "seed": seed,
                    "rank": rank,
                    "temperature": temperature,
                    "batch": batch,
                    "group_size": group_size,
                    "lr": lr,
                    "steps": steps,
                    "wave": "6-sensitivity",
                    "sweep": exp["sweep"],
                },
                reinit=True,
            )
        except Exception as e:
            print(f"  [{tag}] W&B init failed: {e}", flush=True)

        # Tinker client — LoRA rank varies here
        svc = tinker.ServiceClient()
        tc = svc.create_lora_training_client(
            base_model=MODEL, rank=rank, seed=seed
        )
        run_id = tc.model_id

        w0 = tc.save_weights_for_sampler(name="s0").result()
        sc = tc.create_sampling_client(model_path=w0.path)

        _advs = []

        def loss_fn(data, lp):
            losses = [(-_advs[i] * lp[i].sum()) for i in range(len(lp))]
            loss = torch.stack(losses).mean()
            return loss, {"loss": loss.item()}

        step_rewards, step_log = [], []
        examples = list(gsm8k)
        random.shuffle(examples)

        for step in range(steps):
            # batch controls how many GSM8K prompts per GRPO step
            batch_examples = random.sample(examples, min(batch, len(examples)))
            all_data, all_advs, batch_r = [], [], []
            zero_var_prompts = 0

            for question, ans in batch_examples:
                prompt = (
                    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                    f"<|im_start|>user\n{question + QUESTION_SUFFIX}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                pid = tok.encode(prompt, add_special_tokens=False)
                if len(pid) > 1024:
                    pid = pid[:1024]

                sp = T.SamplingParams(
                    max_tokens=512, temperature=temperature, top_p=0.95
                )
                resp = sc.sample(
                    T.ModelInput.from_ints(pid),
                    num_samples=group_size,
                    sampling_params=sp,
                ).result()

                rews = [
                    reward_fn(
                        tok.decode(list(r.tokens), skip_special_tokens=True),
                        ans,
                    )
                    for r in resp.sequences
                ]
                mr = sum(rews) / len(rews)
                sr = (
                    sum((r - mr) ** 2 for r in rews) / len(rews)
                ) ** 0.5 + 1e-8
                if (
                    sum((r - mr) ** 2 for r in rews) / max(len(rews), 1)
                ) < 1e-10:
                    zero_var_prompts += 1
                advs = [(r - mr) / sr for r in rews]
                batch_r.extend(rews)

                for r_seq, adv in zip(resp.sequences, advs):
                    rid = list(r_seq.tokens)
                    fid = pid + rid
                    tid = fid[1:] + [0]
                    all_data.append(
                        T.Datum(
                            model_input=T.ModelInput.from_ints(fid),
                            loss_fn_inputs={
                                "target_tokens": T.TensorData(
                                    data=tid,
                                    dtype="int64",
                                    shape=[len(tid)],
                                )
                            },
                        )
                    )
                    all_advs.append(adv)

            if not all_data:
                continue

            _advs.clear()
            _advs.extend(all_advs)
            fwdbwd = tc.forward_backward_custom(
                data=all_data, loss_fn=loss_fn
            ).result()
            tc.optim_step(
                T.AdamParams(
                    learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8
                )
            ).result()

            avg = sum(batch_r) / len(batch_r) if batch_r else 0.0
            step_rewards.append(avg)

            loss_val = None
            try:
                loss_val = float(fwdbwd.metrics.get("loss", 0.0))
            except Exception:
                pass

            zvf = zero_var_prompts / max(len(batch_examples), 1)
            step_log.append(
                {
                    "step": step + 1,
                    "reward": avg,
                    "loss": loss_val,
                    "zvf": zvf,
                    "gu": 1.0 - zvf,
                }
            )

            if step % 5 == 0:
                print(
                    f"  [{tag}] step {step+1}/{steps} reward={avg:.3f}",
                    flush=True,
                )

            if wb_run:
                try:
                    wb_run.log(
                        {
                            "train/reward": avg,
                            "train/step": step + 1,
                            "train/loss": loss_val,
                            "train/zvf": zvf,
                        }
                    )
                except Exception:
                    pass

            # Refresh sampler every 10 steps (as campaign_v2)
            if (step + 1) % 10 == 0:
                ckpt = tc.save_weights_for_sampler(
                    name=f"s{step+1}"
                ).result()
                sc = tc.create_sampling_client(model_path=ckpt.path)

        final_ckpt = tc.save_weights_for_sampler(name="final").result()

        peak = max(step_rewards) if step_rewards else 0.0
        last10 = (
            sum(step_rewards[-10:]) / min(10, len(step_rewards))
            if step_rewards
            else 0.0
        )
        first5 = (
            sum(step_rewards[:5]) / min(5, len(step_rewards))
            if step_rewards
            else 0.0
        )
        zero_reward_pct = (
            100.0 * sum(1 for r in step_rewards if r == 0.0) / len(step_rewards)
            if step_rewards
            else 0.0
        )
        zero_loss_pct = (
            100.0
            * sum(
                1
                for s in step_log
                if s["loss"] is not None and abs(s["loss"]) < 1e-9
            )
            / max(len(step_log), 1)
        )

        result = {
            "tag": tag,
            "sweep": exp["sweep"],
            "model": MODEL,
            "model_short": MODEL_SHORT,
            "task": "gsm8k",
            "status": "completed",
            "seed": seed,
            "rank": rank,
            "temperature": temperature,
            "batch": batch,
            "group_size": group_size,
            "lr": lr,
            "steps": steps,
            "run_id": run_id,
            "checkpoint": final_ckpt.path,
            "peak": peak,
            "peak_reward": peak,
            "last10_avg": last10,
            "first5_avg": first5,
            "zero_reward_pct": zero_reward_pct,
            "zero_loss_pct": zero_loss_pct,
            "reward_trace": step_rewards,
            "step_log": step_log,
            "wall_clock_sec": time.time() - t0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        try:
            if wb_run:
                wb_run.summary.update(
                    {
                        "peak_reward": peak,
                        "last10_avg": last10,
                        "first5_avg": first5,
                    }
                )
                wb_run.finish()
        except Exception:
            pass

        print(
            f"  ✓ [{tag}] DONE peak={peak:.3f} last10={last10:.3f} "
            f"({result['wall_clock_sec']:.0f}s)",
            flush=True,
        )
        return result

    except Exception as e:
        print(f"  ✗ [{tag}] FAILED: {e}", flush=True)
        traceback.print_exc()
        try:
            if wb_run:
                wb_run.finish(exit_code=1)
        except Exception:
            pass
        return {
            "tag": tag,
            "sweep": exp["sweep"],
            "model": MODEL,
            "model_short": MODEL_SHORT,
            "seed": seed,
            "rank": rank,
            "temperature": temperature,
            "batch": batch,
            "status": "failed",
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ── Experiment list (with baseline de-duplication) ─────────────────
def build_experiments():
    exps = []

    # Baseline — appears in all three sweeps
    baseline_tag = f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
    exps.append(
        {
            "tag": baseline_tag,
            "sweep": "baseline",
            "rank": BASE_RANK,
            "temperature": BASE_TEMP,
            "batch": BASE_BATCH,
        }
    )

    # Temperature sweep — vary temperature, fix rank+batch
    for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
        if t == BASE_TEMP:
            continue  # covered by baseline
        exps.append(
            {
                "tag": f"w6_temp{t}_r{BASE_RANK}_b{BASE_BATCH}",
                "sweep": "temperature",
                "rank": BASE_RANK,
                "temperature": t,
                "batch": BASE_BATCH,
            }
        )

    # LoRA rank sweep — vary rank, fix temp+batch
    for r in [4, 8, 16, 32, 64]:
        if r == BASE_RANK:
            continue
        exps.append(
            {
                "tag": f"w6_rank{r}_t{BASE_TEMP}_b{BASE_BATCH}",
                "sweep": "rank",
                "rank": r,
                "temperature": BASE_TEMP,
                "batch": BASE_BATCH,
            }
        )

    # Batch sweep — vary batch, fix rank+temp
    for b in [1, 2, 4, 8]:
        if b == BASE_BATCH:
            continue
        exps.append(
            {
                "tag": f"w6_batch{b}_r{BASE_RANK}_t{BASE_TEMP}",
                "sweep": "batch",
                "rank": BASE_RANK,
                "temperature": BASE_TEMP,
                "batch": b,
            }
        )

    return exps


def launch(max_parallel=6):
    exps = build_experiments()
    print(
        f"\n{'='*70}\n"
        f"WAVE 6 — Qwen3-8B Sensitivity Ablations ({len(exps)} experiments)\n"
        f"Max parallel: {max_parallel}\n"
        f"{'='*70}\n",
        flush=True,
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        futures = {ex.submit(run_one, e): e for e in exps}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                exp = futures[fut]
                results.append(
                    {
                        "tag": exp["tag"],
                        "sweep": exp["sweep"],
                        "status": "crashed",
                        "error": str(e),
                    }
                )
            # Incremental save
            _save(results, exps)

    _save(results, exps, final=True)
    return results


def _save(results, exps, final=False):
    # Organize for the results file
    by_tag = {r["tag"]: r for r in results}

    def _collect(sweep_name, key, values):
        rows = []
        for v in values:
            if sweep_name == "temperature":
                tag = (
                    f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                    if v == BASE_TEMP
                    else f"w6_temp{v}_r{BASE_RANK}_b{BASE_BATCH}"
                )
            elif sweep_name == "rank":
                tag = (
                    f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                    if v == BASE_RANK
                    else f"w6_rank{v}_t{BASE_TEMP}_b{BASE_BATCH}"
                )
            else:  # batch
                tag = (
                    f"w6_base_r{BASE_RANK}_t{BASE_TEMP}_b{BASE_BATCH}"
                    if v == BASE_BATCH
                    else f"w6_batch{v}_r{BASE_RANK}_t{BASE_TEMP}"
                )
            r = by_tag.get(tag)
            if r:
                rows.append(
                    {
                        key: v,
                        **{
                            k: r.get(k)
                            for k in [
                                "status",
                                "peak_reward",
                                "last10_avg",
                                "first5_avg",
                                "zero_reward_pct",
                                "zero_loss_pct",
                                "reward_trace",
                                "step_log",
                                "run_id",
                                "checkpoint",
                                "seed",
                                "rank",
                                "temperature",
                                "batch",
                                "group_size",
                                "lr",
                                "steps",
                                "wall_clock_sec",
                                "tag",
                            ]
                        },
                    }
                )
            else:
                rows.append({key: v, "status": "pending", "tag": tag})
        return rows

    out = {
        "metadata": {
            "wave": 6,
            "title": "Qwen3-8B GSM8K Sensitivity Ablations",
            "model": MODEL,
            "task": "gsm8k",
            "seed": SEED,
            "steps": STEPS,
            "group_size": GROUP_SIZE,
            "lr": LR,
            "baseline": {
                "rank": BASE_RANK,
                "temperature": BASE_TEMP,
                "batch": BASE_BATCH,
            },
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(exps),
            "completed": sum(
                1 for r in results if r.get("status") == "completed"
            ),
            "failed": sum(
                1 for r in results if r.get("status") in ("failed", "crashed")
            ),
            "finalized": final,
            "wandb_project": (
                "https://wandb.ai/arvindcr4-pes-university/"
                "tinker-rl-lab-world-class"
            ),
        },
        "temperature_sweep": _collect(
            "temperature", "temperature", [0.2, 0.4, 0.6, 0.8, 1.0]
        ),
        "rank_sweep": _collect("rank", "rank", [4, 8, 16, 32, 64]),
        "batch_sweep": _collect("batch", "batch", [1, 2, 4, 8]),
        "runs": results,
    }

    out_dir = os.path.dirname(RESULTS_PATH)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    max_par = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    launch(max_parallel=max_par)
