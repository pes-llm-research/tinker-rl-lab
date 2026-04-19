#!/usr/bin/env python3
"""
Held-out GSM8K evaluation of the top-10 Tinker checkpoints.

Selection: the 10 Tinker run checkpoints with the highest training-time
``last10_avg`` reward across the repo's result files.

Evaluation: for each checkpoint, sample greedily (T=0) on a fixed, deterministic
held-out slice of the GSM8K ``test`` split (disjoint from the training split
used by every campaign run). We score with the same boxed-answer reward used
during training, so held-out accuracy is directly comparable to the reported
training last-10 metric.

Designed to run on Modal (``modal run modal_heldout_eval.py``) which parallelises
checkpoints across containers, but it is self-contained and also runs locally
via ``python modal_heldout_eval.py --local``.

Outputs:
  experiments/results/heldout_gsm8k.json

Env:
  TINKER_API_KEY  required (sampling goes through the Tinker service; the
                  Modal container only orchestrates + scores)
  WANDB_API_KEY   optional (logs a summary run to
                  arvindcr4-pes-university/tinker-rl-lab-world-class)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "heldout_gsm8k.json"
MASTER_RESULTS = EXPERIMENTS_DIR / "master_results.json"
TINKER_RESULTS_DIR = EXPERIMENTS_DIR / "tinker-runs" / "results"
CONSOLIDATED = EXPERIMENTS_DIR / "all_results_consolidated.json"

# ---------------------------------------------------------------------------
# Evaluation constants
# ---------------------------------------------------------------------------
# Same system / suffix as campaign_v2.py so held-out scoring matches training.
SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step, then give your "
    "final numerical answer inside \\boxed{}."
)
QUESTION_SUFFIX = (
    " Provide a numerical answer without units, written inside \\boxed{}."
)

HELDOUT_N = 500                 # fixed held-out size (§5 of the paper)
HELDOUT_SEED = 0                # deterministic selection
MAX_TOKENS = 512
TEMPERATURE = 0.0               # greedy decoding for eval
TOP_P = 1.0
PROMPT_MAX_LEN = 1024           # same cap as training
EVAL_MAX_WORKERS = 8            # concurrent sample calls per checkpoint

WANDB_PROJECT = "tinker-rl-lab-world-class"
WANDB_ENTITY = "arvindcr4-pes-university"


# ---------------------------------------------------------------------------
# Reward function (identical to campaign_v2.reward_fn)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Top-10 checkpoint discovery
# ---------------------------------------------------------------------------
CKPT_RE = re.compile(r"^tinker://[a-f0-9\-]+:[a-z0-9]+:\d+/sampler_weights/[a-z0-9]+$")


@dataclass
class RunRecord:
    checkpoint: str
    model: str | None = None
    model_short: str | None = None
    seed: int | None = None
    last10_avg: float = 0.0
    peak: float | None = None
    steps: int | None = None
    group_size: int | None = None
    lr: float | None = None
    run_id: str | None = None
    task: str = "gsm8k"
    source_file: str | None = None
    wandb_run_url: str | None = None
    experiment_id: str | None = None


def _as_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _walk_for_runs(obj: Any, src: str, acc: list[RunRecord]) -> None:
    if isinstance(obj, dict):
        ck = (
            obj.get("checkpoint")
            or obj.get("hf_checkpoint_url")
            or obj.get("final_checkpoint")
        )
        l10 = obj.get("last10_avg")
        if isinstance(ck, str) and CKPT_RE.match(ck) and _as_float(l10) is not None:
            acc.append(
                RunRecord(
                    checkpoint=ck,
                    model=obj.get("model"),
                    model_short=obj.get("model_short"),
                    seed=obj.get("seed"),
                    last10_avg=float(l10),
                    peak=_as_float(obj.get("peak") or obj.get("peak_reward")),
                    steps=obj.get("steps") or obj.get("steps_completed"),
                    group_size=obj.get("group") or obj.get("group_size"),
                    lr=_as_float(obj.get("lr")),
                    run_id=obj.get("run_id"),
                    task=obj.get("task") or "gsm8k",
                    source_file=src,
                    wandb_run_url=obj.get("wandb_run_url"),
                    experiment_id=obj.get("experiment")
                    or obj.get("experiment_id")
                    or obj.get("tag")
                    or obj.get("name"),
                )
            )
        for v in obj.values():
            _walk_for_runs(v, src, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk_for_runs(v, src, acc)


def discover_top_checkpoints(
    k: int = 10,
    task: str = "gsm8k",
    extra_paths: Iterable[Path] = (),
) -> list[RunRecord]:
    """Find the top-k unique Tinker checkpoints by training last10_avg."""
    search_paths: list[Path] = [MASTER_RESULTS, CONSOLIDATED]
    search_paths.extend(sorted(TINKER_RESULTS_DIR.glob("*.json")))
    search_paths.extend(extra_paths)

    rows: list[RunRecord] = []
    for p in search_paths:
        if not p or not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        _walk_for_runs(data, str(p.relative_to(REPO_ROOT)), rows)

    # Filter to task; prefer records that name an explicit model over anonymous
    rows = [r for r in rows if (r.task or "gsm8k") == task]
    # Dedup by checkpoint, keeping the record with the most filled-in fields.
    best: dict[str, RunRecord] = {}
    for r in rows:
        cur = best.get(r.checkpoint)
        if cur is None:
            best[r.checkpoint] = r
            continue
        score = lambda rr: sum(v is not None for v in rr.__dict__.values())  # noqa: E731
        if score(r) > score(cur):
            best[r.checkpoint] = r
    ranked = sorted(best.values(), key=lambda r: r.last10_avg, reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Dataset: held-out GSM8K slice
# ---------------------------------------------------------------------------
def load_heldout_gsm8k(n: int = HELDOUT_N, seed: int = HELDOUT_SEED):
    """Load a deterministic n-problem slice of the GSM8K ``test`` split.

    The campaign runs train on the GSM8K ``train`` split, so the entire ``test``
    split (1319 problems) is strictly held out. We shuffle with a fixed seed and
    take the first ``n`` to get a stable, reproducible subset for reporting.
    """
    from datasets import load_dataset  # local import: not needed for discovery

    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems: list[tuple[str, str]] = []
    for row in ds:
        m = re.search(r"####\s*([\-\d,\.]+)", row["answer"])
        if not m:
            continue
        ans = m.group(1).replace(",", "").strip()
        problems.append((row["question"], ans))

    rng = random.Random(seed)
    idx = list(range(len(problems)))
    rng.shuffle(idx)
    selected = [problems[i] for i in idx[:n]]
    return selected


# ---------------------------------------------------------------------------
# Per-checkpoint evaluation (shared by local + Modal entrypoints)
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    checkpoint: str
    model: str
    model_short: str | None
    seed: int | None
    training_last10_avg: float
    heldout_n: int
    heldout_correct: int
    heldout_accuracy: float
    ci95_low: float
    ci95_high: float
    mean_completion_tokens: float
    elapsed_seconds: float
    run_id: str | None
    source_file: str | None
    experiment_id: str | None
    wandb_run_url: str | None
    per_problem: list[dict] = field(default_factory=list)


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


_TOKENIZER_CACHE: dict[str, Any] = {}
import threading as _threading
_TOKENIZER_LOCK = _threading.Lock()


# If the primary repo is gated and no HF_TOKEN is available we fall back to
# a public mirror with an identical tokenizer. Tinker does the model weight
# lookup server-side, so only the local tokenization depends on HF access.
_TOKENIZER_FALLBACKS = {
    "meta-llama/Llama-3.1-8B-Instruct": "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B": "NousResearch/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B": "NousResearch/Meta-Llama-3.1-70B",
    "meta-llama/Llama-3.3-70B-Instruct": "unsloth/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.2-1B": "unsloth/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B": "unsloth/Llama-3.2-3B",
}


def _get_tokenizer(model: str):
    """Thread-safe, process-wide tokenizer cache.

    transformers has a lazy-import hook that is not race-safe under threads,
    and we intentionally run many checkpoints in parallel, so we serialise the
    first import and memoise per-model tokenizers here.
    """
    tok = _TOKENIZER_CACHE.get(model)
    if tok is not None:
        return tok
    with _TOKENIZER_LOCK:
        tok = _TOKENIZER_CACHE.get(model)
        if tok is not None:
            return tok
        from transformers import AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        except Exception as e:
            fallback = _TOKENIZER_FALLBACKS.get(model)
            if not fallback:
                raise
            print(
                f"[heldout] tokenizer load failed for {model} ({e.__class__.__name__}); "
                f"falling back to {fallback}",
                flush=True,
            )
            tok = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
        _TOKENIZER_CACHE[model] = tok
    return tok


def evaluate_checkpoint(
    record: dict,
    problems: list[tuple[str, str]],
    max_workers: int = EVAL_MAX_WORKERS,
    keep_traces: bool = False,
) -> dict:
    """Greedy-sample every held-out problem through the given checkpoint and score."""
    import tinker
    import tinker.types as T

    rec = RunRecord(**record) if not isinstance(record, RunRecord) else record
    model = rec.model
    if not model:
        raise ValueError(f"checkpoint {rec.checkpoint} is missing a model name")

    print(f"[eval] {model} seed={rec.seed} last10={rec.last10_avg:.3f}", flush=True)
    print(f"       {rec.checkpoint}", flush=True)

    svc = tinker.ServiceClient()
    sc = svc.create_sampling_client(model_path=rec.checkpoint)
    tok = _get_tokenizer(model)

    sp = T.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P)

    def build_prompt(question: str) -> list[int]:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{question + QUESTION_SUFFIX}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        pid = tok.encode(prompt, add_special_tokens=False)
        if len(pid) > PROMPT_MAX_LEN:
            pid = pid[:PROMPT_MAX_LEN]
        return pid

    def score_one(idx: int, question: str, answer: str) -> dict:
        pid = build_prompt(question)
        resp = sc.sample(
            T.ModelInput.from_ints(pid), num_samples=1, sampling_params=sp
        ).result()
        seq = resp.sequences[0]
        toks = list(seq.tokens)
        text = tok.decode(toks, skip_special_tokens=True)
        r = reward_fn(text, answer)
        return {
            "idx": idx,
            "answer": answer,
            "reward": r,
            "completion_tokens": len(toks),
            "completion": text if keep_traces else None,
        }

    t0 = time.time()
    per_problem: list[dict] = [None] * len(problems)  # type: ignore[list-item]
    correct = 0
    tok_total = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(score_one, i, q, a): i for i, (q, a) in enumerate(problems)
        }
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                out = {
                    "idx": i,
                    "answer": problems[i][1],
                    "reward": 0.0,
                    "completion_tokens": 0,
                    "completion": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            per_problem[i] = out
            correct += int(out["reward"] >= 1.0)
            tok_total += out.get("completion_tokens", 0)
            done += 1
            if done % 50 == 0 or done == len(problems):
                print(
                    f"       progress {done}/{len(problems)}  "
                    f"acc={correct/done:.3f}",
                    flush=True,
                )

    elapsed = time.time() - t0
    n = len(problems)
    acc = correct / n if n else 0.0
    lo, hi = _wilson_ci(correct, n)

    result = EvalResult(
        checkpoint=rec.checkpoint,
        model=model,
        model_short=rec.model_short,
        seed=rec.seed,
        training_last10_avg=rec.last10_avg,
        heldout_n=n,
        heldout_correct=correct,
        heldout_accuracy=acc,
        ci95_low=lo,
        ci95_high=hi,
        mean_completion_tokens=(tok_total / n) if n else 0.0,
        elapsed_seconds=elapsed,
        run_id=rec.run_id,
        source_file=rec.source_file,
        experiment_id=rec.experiment_id,
        wandb_run_url=rec.wandb_run_url,
        per_problem=(
            per_problem
            if keep_traces
            else [
                {k: v for k, v in p.items() if k != "completion"} for p in per_problem
            ]
        ),
    )
    print(
        f"[eval] ✓ {model} seed={rec.seed}  acc={acc:.3f} "
        f"({correct}/{n}) CI95=[{lo:.3f},{hi:.3f}]  {elapsed:.0f}s",
        flush=True,
    )
    return asdict(result)


# ---------------------------------------------------------------------------
# Modal entrypoint
# ---------------------------------------------------------------------------
try:
    import modal

    _MODAL_AVAILABLE = True
except Exception:
    _MODAL_AVAILABLE = False


def _build_modal_secrets() -> list:
    """Attach Tinker (required) + W&B / HF (optional) Modal secrets."""
    secrets: list = [
        modal.Secret.from_name("tinker-api", required_keys=["TINKER_API_KEY"])
    ]
    for name, keys in (("wandb", ["WANDB_API_KEY"]), ("huggingface", ["HF_TOKEN"])):
        try:
            secrets.append(modal.Secret.from_name(name, required_keys=keys))
        except Exception as e:  # pragma: no cover - optional
            print(f"[heldout] optional Modal secret '{name}' unavailable: {e}")
    return secrets


# ---------------------------------------------------------------------------
# Shared runner (used by both modal local_entrypoint and --local CLI mode)
# ---------------------------------------------------------------------------
def run(
    k: int = 10,
    n: int = HELDOUT_N,
    seed: int = HELDOUT_SEED,
    output: Path = DEFAULT_OUTPUT,
    dry_run: bool = False,
    remote_fn=None,
) -> Path:
    """Orchestrate checkpoint selection → evaluation → aggregation → persistence."""
    # Warm up the transformers lazy-import hook on the main thread so parallel
    # worker threads below don't race on it.
    try:
        import transformers as _transformers  # noqa: F401
        from transformers import AutoTokenizer as _AT  # noqa: F401
    except Exception:
        pass
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"[heldout] selecting top-{k} checkpoints by training last10_avg…")
    top = discover_top_checkpoints(k=k, task="gsm8k")
    if len(top) < k:
        print(
            f"[heldout] WARNING: only found {len(top)} unique checkpoints "
            f"with last10_avg metadata (requested {k})."
        )
    for i, r in enumerate(top, 1):
        print(
            f"  {i:2d}. last10={r.last10_avg:.4f}  {r.model}  seed={r.seed}  "
            f"{r.checkpoint}"
        )

    if dry_run:
        print("[heldout] --dry-run: stopping before evaluation")
        return output

    print(f"[heldout] loading held-out GSM8K slice  n={n}  seed={seed}")
    problems = load_heldout_gsm8k(n=n, seed=seed)
    print(f"[heldout] loaded {len(problems)} problems from GSM8K test split")

    records = [asdict(r) for r in top]

    results: list[dict] = []
    t0 = time.time()
    if remote_fn is not None:
        # Modal: map across checkpoints in parallel containers.
        # ``eval_one(record, problems)`` takes two positional args; Modal's
        # ``.map`` expects one iterable per function argument (it does not
        # auto-unpack tuples). We broadcast the shared ``problems`` list once
        # per checkpoint so each remote call gets ``(record, problems)``.
        print("[heldout] dispatching to Modal…")
        for res in remote_fn(records, [problems] * len(records)):
            results.append(res)
    else:
        # Local: run checkpoints in parallel threads. Sampling is I/O-bound
        # (it all happens on the Tinker service) so threads are plenty.
        max_ckpt_workers = int(os.environ.get("HELDOUT_CKPT_WORKERS", "10"))
        per_ckpt_workers = int(
            os.environ.get("HELDOUT_PROBLEM_WORKERS", str(EVAL_MAX_WORKERS))
        )
        print(
            f"[heldout] local mode: {max_ckpt_workers} checkpoints × "
            f"{per_ckpt_workers} problem workers in parallel"
        )

        def _safe_eval(rec: dict) -> dict:
            try:
                return evaluate_checkpoint(
                    rec, problems, max_workers=per_ckpt_workers, keep_traces=False
                )
            except Exception as e:
                print(f"[heldout] ✗ {rec.get('model')} failed: {e}", flush=True)
                return {
                    **rec,
                    "error": f"{type(e).__name__}: {e}",
                    "heldout_n": 0,
                    "heldout_correct": 0,
                    "heldout_accuracy": 0.0,
                }

        with ThreadPoolExecutor(max_workers=max_ckpt_workers) as ex:
            futs = [ex.submit(_safe_eval, rec) for rec in records]
            for fut in as_completed(futs):
                results.append(fut.result())

    # Preserve the training ``last10_avg`` ranking used for selection (docs
    # and paper refer to this as the "top-k by last-10" order). We attach a
    # secondary ``heldout_rank`` field per record so downstream consumers can
    # still obtain the held-out-accuracy ordering without mutating the JSON
    # order.
    heldout_rank = {
        id(r): i + 1
        for i, r in enumerate(
            sorted(results, key=lambda r: r.get("heldout_accuracy", 0.0), reverse=True)
        )
    }
    for r in results:
        r["heldout_rank"] = heldout_rank.get(id(r))

    # Aggregate summary. ``heldout_n`` reports the actual number of held-out
    # problems evaluated (``len(problems)``) rather than the requested ``n``
    # so the metadata stays consistent with per-checkpoint ``heldout_n`` when
    # ``load_heldout_gsm8k`` skips rows or ``n`` exceeds the available split.
    valid = [r for r in results if r.get("heldout_n")]
    actual_n = len(problems)
    summary = {
        "n_checkpoints": len(results),
        "heldout_n": actual_n,
        "heldout_n_requested": n,
        "heldout_seed": seed,
        "mean_heldout_accuracy": (
            sum(r["heldout_accuracy"] for r in valid) / len(valid) if valid else 0.0
        ),
        "best_heldout_accuracy": max(
            (r["heldout_accuracy"] for r in valid), default=0.0
        ),
        "total_elapsed_seconds": time.time() - t0,
    }

    payload = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "description": (
                "Held-out GSM8K evaluation of the top-10 Tinker checkpoints "
                "(ranked by training last10_avg)."
            ),
            "heldout_split": "openai/gsm8k[test]",
            "heldout_n": actual_n,
            "heldout_n_requested": n,
            "heldout_seed": seed,
            "system_prompt": SYSTEM_PROMPT,
            "question_suffix": QUESTION_SUFFIX,
            "sampling": {
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
            },
            "selection_sources": [
                str(MASTER_RESULTS.relative_to(REPO_ROOT)),
                str(CONSOLIDATED.relative_to(REPO_ROOT)),
                str(TINKER_RESULTS_DIR.relative_to(REPO_ROOT)) + "/*.json",
            ],
            "wandb_project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        },
        "summary": summary,
        "results": results,
    }

    output.write_text(json.dumps(payload, indent=2))
    print(f"[heldout] wrote {output.relative_to(REPO_ROOT)}")

    # Optional: log summary to W&B
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb

            run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=f"heldout_gsm8k_top{k}_{datetime.utcnow():%Y%m%d_%H%M%S}",
                tags=["heldout-eval", "gsm8k", f"n={n}"],
                config={"k": k, "heldout_n": n, "heldout_seed": seed},
                reinit=True,
            )
            table = wandb.Table(
                columns=[
                    "model",
                    "seed",
                    "training_last10_avg",
                    "heldout_accuracy",
                    "ci95_low",
                    "ci95_high",
                    "heldout_n",
                    "checkpoint",
                ]
            )
            for r in results:
                table.add_data(
                    r.get("model"),
                    r.get("seed"),
                    r.get("training_last10_avg"),
                    r.get("heldout_accuracy"),
                    r.get("ci95_low"),
                    r.get("ci95_high"),
                    r.get("heldout_n"),
                    r.get("checkpoint"),
                )
            run.log({"heldout/table": table, **summary})
            run.summary.update(summary)
            run.finish()
        except Exception as e:  # pragma: no cover - non-critical
            print(f"[heldout] W&B logging skipped: {e}")

    return output


# ---------------------------------------------------------------------------
# Modal app (module-level so `modal run` can discover `app` and `main`).
# If modal isn't installed we silently skip — the module still works as a
# plain CLI via `python modal_heldout_eval.py --local`.
# ---------------------------------------------------------------------------
if _MODAL_AVAILABLE and os.environ.get("MODAL_EVAL_DISABLE_APP") != "1":
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "tinker>=0.18.0",
            "transformers>=4.40",
            "datasets",
            "huggingface_hub",
            "wandb",
            "numpy",
        )
        .env({"WANDB_PROJECT": WANDB_PROJECT, "WANDB_ENTITY": WANDB_ENTITY})
    )
    app = modal.App("tinker-rl-heldout-eval", image=image)

    try:
        _MODAL_SECRETS = _build_modal_secrets()
    except Exception as e:  # pragma: no cover - keeps module importable
        print(f"[heldout] Modal secrets unavailable ({e}); falling back to env")
        _MODAL_SECRETS = []

    @app.function(timeout=60 * 60, secrets=_MODAL_SECRETS, retries=1)
    def eval_one(record: dict, problems: list) -> dict:
        return evaluate_checkpoint(record, problems, keep_traces=False)

    @app.local_entrypoint()
    def main(
        k: int = 10,
        n: int = HELDOUT_N,
        seed: int = HELDOUT_SEED,
        output: str = str(DEFAULT_OUTPUT),
        dry_run: bool = False,
    ):
        """Modal local entrypoint: fan out checkpoints across containers."""
        run(
            k=k,
            n=n,
            seed=seed,
            output=Path(output),
            dry_run=dry_run,
            remote_fn=eval_one.map,
        )


# ---------------------------------------------------------------------------
# Local CLI entrypoint
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=10, help="number of top checkpoints")
    p.add_argument("--n", type=int, default=HELDOUT_N, help="held-out problems")
    p.add_argument("--seed", type=int, default=HELDOUT_SEED)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--local",
        action="store_true",
        help="evaluate sequentially in-process instead of dispatching to Modal",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="only print the top-k selection, do not evaluate",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.local or not _MODAL_AVAILABLE:
        run(
            k=args.k,
            n=args.n,
            seed=args.seed,
            output=args.output,
            dry_run=args.dry_run,
            remote_fn=None,
        )
    else:
        print(
            "[heldout] Modal detected — run `modal run experiments/modal/"
            "modal_heldout_eval.py` to parallelise, or pass --local for "
            "in-process eval."
        )
        print("[heldout] Falling through to --local mode for this direct invocation.")
        run(
            k=args.k,
            n=args.n,
            seed=args.seed,
            output=args.output,
            dry_run=args.dry_run,
            remote_fn=None,
        )
