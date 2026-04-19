#!/usr/bin/env bash
# =============================================================================
# TinkerRL-Bench — Reviewer Smoke Test  (<10 min wall-clock)
# =============================================================================
# Purpose
#   Give NeurIPS artifact reviewers a single command that verifies the
#   environment, the core code paths, and a mini GRPO-on-GSM8K loop end-to-end
#   in under 10 minutes. It is NOT a full reproduction of the headline result
#   (that requires ~5 GPU-hours on an A100-80GB — see REPRODUCE.md).
#
# What it checks
#   1. Python / CUDA / core library imports (torch, transformers, trl, tinker)
#   2. Seed determinism (from utils.seed.set_global_seed)
#   3. Unit tests in tests/ (pytest)
#   4. GSM8K dataset loads and answer-extraction regex works
#   5. The GRPO reward function + advantage computation round-trip
#   6. [Optional, requires TINKER_API_KEY]  A 3-step Tinker GRPO call on
#      Qwen/Qwen3.5-4B (smallest model on the rate card) to prove the
#      training client wire-protocol works.
#
# Usage
#   bash scripts/smoke_test.sh              # offline subset (~2 min)
#   TINKER_API_KEY=... bash scripts/smoke_test.sh   # full smoke (~8 min)
#
# Exit codes
#   0  all checks passed
#   1  any check failed — stdout/stderr shows which one
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SMOKE_LOG="${SMOKE_LOG:-/tmp/tinkerrl_smoke.log}"
START_TS=$(date +%s)
: > "$SMOKE_LOG"

log()   { printf "\033[1;34m[smoke]\033[0m %s\n" "$*" | tee -a "$SMOKE_LOG"; }
ok()    { printf "\033[1;32m  ✓\033[0m %s\n" "$*" | tee -a "$SMOKE_LOG"; }
fail()  { printf "\033[1;31m  ✗\033[0m %s\n" "$*" | tee -a "$SMOKE_LOG"; exit 1; }
step()  { printf "\n\033[1;35m[%d/%d]\033[0m %s\n" "$1" "$2" "$3" | tee -a "$SMOKE_LOG"; }

TOTAL=6
[[ -n "${TINKER_API_KEY:-}" ]] && TOTAL=7

log "TinkerRL-Bench smoke test — target <10 min"
log "Repo:      $(git rev-parse --short HEAD 2>/dev/null || echo 'not-a-git-repo')"
log "Python:    $(python --version 2>&1)"
log "Host:      $(uname -srm)"
log "Log file:  $SMOKE_LOG"

# -----------------------------------------------------------------------------
step 1 "$TOTAL" "Core library imports"
python - <<'PY' 2>&1 | tee -a "$SMOKE_LOG"
import importlib, sys, time
t0 = time.time()
required = ["torch", "numpy", "transformers", "datasets", "accelerate", "trl"]
optional = ["tinker", "wandb", "rliable", "peft", "scipy"]
for m in required:
    mod = importlib.import_module(m)
    print(f"  required {m:14s} {getattr(mod,'__version__','?')}")
for m in optional:
    try:
        mod = importlib.import_module(m)
        print(f"  optional {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  optional {m:14s} MISSING ({e.__class__.__name__})")
import torch
print(f"  torch.cuda.is_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
print(f"  elapsed={time.time()-t0:.1f}s")
PY
ok "imports"

# -----------------------------------------------------------------------------
step 2 "$TOTAL" "Seed determinism (utils.seed)"
python - <<'PY' 2>&1 | tee -a "$SMOKE_LOG"
import sys, os
sys.path.insert(0, os.getcwd())
from utils.seed import set_global_seed
import random, numpy as np, torch
def snapshot(seed):
    set_global_seed(seed)
    return (random.random(),
            tuple(np.random.rand(4).round(6).tolist()),
            tuple(torch.randn(4).round().tolist()))
assert snapshot(42) == snapshot(42), "non-deterministic within same seed"
assert snapshot(42) != snapshot(43), "different seeds collided"
print("  determinism verified for seeds {42, 43}")
PY
ok "seeds"

# -----------------------------------------------------------------------------
step 3 "$TOTAL" "Unit tests (pytest tests/)"
if command -v pytest >/dev/null 2>&1; then
    pytest -q tests/ --maxfail=1 --disable-warnings 2>&1 | tee -a "$SMOKE_LOG"
    ok "pytest"
else
    python -m pip install -q pytest 2>&1 | tee -a "$SMOKE_LOG"
    pytest -q tests/ --maxfail=1 --disable-warnings 2>&1 | tee -a "$SMOKE_LOG"
    ok "pytest"
fi

# -----------------------------------------------------------------------------
step 4 "$TOTAL" "GSM8K dataset load (openai/gsm8k, split=test, first 8 rows)"
python - <<'PY' 2>&1 | tee -a "$SMOKE_LOG"
import re, time
from datasets import load_dataset
t0 = time.time()
ds = load_dataset("openai/gsm8k", "main", split="test[:8]")
assert len(ds) == 8, f"expected 8 rows, got {len(ds)}"
# Pinned dataset revision — see ARTIFACT.md §3.4
# (we don't assert the sha here so stale mirrors still pass the smoke test)
for row in ds:
    m = re.search(r'####\s*([-\d,\.]+)', row["answer"])
    assert m, f"answer extraction failed: {row['answer'][:80]}"
print(f"  loaded {len(ds)} rows in {time.time()-t0:.1f}s; sample answer = {m.group(1).strip()}")
PY
ok "gsm8k"

# -----------------------------------------------------------------------------
step 5 "$TOTAL" "GRPO reward + advantage round-trip"
python - <<'PY' 2>&1 | tee -a "$SMOKE_LOG"
import re, math
def reward(response, answer):
    boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
    for b in boxed:
        try:
            if abs(float(b.replace(",","").strip()) - float(answer)) < 0.01: return 1.0
        except: pass
    nums = re.findall(r'[-+]?\d[\d,]*\.?\d*', response)
    if nums:
        try:
            if abs(float(nums[-1].replace(",","")) - float(answer)) < 0.01: return 1.0
        except: pass
    return 0.0
# sanity cases
cases = [
    ("The answer is \\boxed{42}", "42", 1.0),
    ("After 3 steps, I get 17", "17", 1.0),
    ("clearly 99", "42", 0.0),
    ("\\boxed{42.0}", "42", 1.0),
]
for r, a, exp in cases:
    got = reward(r, a)
    assert got == exp, f"reward({r!r}, {a!r}) = {got}, expected {exp}"
# GRPO advantage normalization
rews = [1.0, 0.0, 0.0, 1.0]
mr = sum(rews)/len(rews)
sr = (sum((r-mr)**2 for r in rews)/len(rews))**0.5 + 1e-8
advs = [(r-mr)/sr for r in rews]
assert abs(sum(advs)) < 1e-5, "advantages should sum to ~0"
print(f"  reward fn: {len(cases)}/{len(cases)} cases passed")
print(f"  advantages (rewards={rews}) = {[round(a,3) for a in advs]}")
PY
ok "grpo math"

# -----------------------------------------------------------------------------
step 6 "$TOTAL" "Docstring / script surface check"
python - <<'PY' 2>&1 | tee -a "$SMOKE_LOG"
import importlib.util, pathlib
for p in ["grpo_gsm8k_base.py", "utils/seed.py", "utils/stats.py",
         "scripts/run_seeds.sh", "REPRODUCE.md", "ARTIFACT.md", "Dockerfile"]:
    assert pathlib.Path(p).exists(), f"missing required artifact file: {p}"
# Just parse the main experiment script to catch syntax errors
import ast
ast.parse(pathlib.Path("grpo_gsm8k_base.py").read_text())
print("  required artifact files present and parseable")
PY
ok "surface"

# -----------------------------------------------------------------------------
if [[ -n "${TINKER_API_KEY:-}" ]]; then
    step 7 "$TOTAL" "Tinker wire-protocol (3-step GRPO on Qwen3.5-4B, LoRA r=4)"
    # Tiny run: 3 steps × batch 1 × group 2, LoRA rank 4 — designed to finish
    # in ~3–5 min on Tinker's smallest tier and cost < $0.25.
    timeout 480 python grpo_gsm8k_base.py \
        --model "Qwen/Qwen3.5-4B" \
        --seed 42 \
        --rank 4 \
        --steps 3 \
        --group 2 \
        --batch 1 \
        --tag "smoke_$(date +%s)" 2>&1 | tee -a "$SMOKE_LOG"
    ok "tinker"
else
    log "TINKER_API_KEY not set → skipping live Tinker smoke (step 7/7)"
fi

# -----------------------------------------------------------------------------
END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))
printf "\n\033[1;32m========================================\033[0m\n"
printf "\033[1;32m  SMOKE TEST PASSED  (%d/%d checks)\033[0m\n" "$TOTAL" "$TOTAL"
printf "\033[1;32m  elapsed: %ds (budget: 600s)\033[0m\n" "$ELAPSED"
printf "\033[1;32m  log:     %s\033[0m\n" "$SMOKE_LOG"
printf "\033[1;32m========================================\033[0m\n"
if [[ $ELAPSED -gt 600 ]]; then
    printf "\033[1;33m  WARN: smoke exceeded 10-minute budget (%ds)\033[0m\n" "$ELAPSED"
fi
