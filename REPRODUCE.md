# Reproducing TinkerRL-Bench

This document gives **reviewers and third parties** exact, copy-pasteable
commands to reproduce every result reported in the NeurIPS 2026 submission
(`paper/main.tex`) and the capstone final report (`reports/capstone_final_report.md`).

The **headline result** reviewers should verify is:

> **GRPO on Qwen3-8B, GSM8K, 30 steps, LoRA rank 32, group size 8**
> Peak accuracy **62.5 %**, last-10 average **34.4 %**
> (paper Table 2, `main.tex:897`; matches W&B run
> [`grpo_qwen3-8b_gsm8k_s123`](https://wandb.ai/arvindcr4-pes-university/tinker-rl-lab-world-class/runs/c1zca2qb)).

Tolerance for reviewer-side reruns is **±5 percentage points** on last-10 and
**±10 percentage points** on peak reward (see §8 below for justification).

---

## 0. TL;DR — one-command reviewer flow

```bash
# 1. Build & enter the pinned container (~6 min cold on a fresh host)
docker build \
  --build-arg GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg GIT_REF=$(git rev-parse --abbrev-ref HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t tinker-rl-lab:repro .

# 2. Under 10 minutes of structural checks — no GPU required
docker run --rm tinker-rl-lab:repro bash scripts/smoke_test.sh

# 3. Full smoke incl. a live 3-step Tinker call (~8 min, < $0.25)
docker run --rm -e TINKER_API_KEY tinker-rl-lab:repro bash scripts/smoke_test.sh

# 4. Headline result — GRPO Qwen3-8B GSM8K (1x A100-80GB, ~5 h wall)
docker run --gpus all --rm \
    -e TINKER_API_KEY -e WANDB_API_KEY -e HF_TOKEN \
    -v $(pwd)/results:/workspace/tinker-rl-lab/results \
    tinker-rl-lab:repro \
    python grpo_gsm8k_base.py \
        --model Qwen/Qwen3-8B --seed 42 --rank 32 \
        --steps 30 --group 8 --batch 2 --lr 1e-5 \
        --tag headline_repro
```

Expected final line from step 4 (within tolerance):

```
[headline_repro] Last-10 avg accuracy: 34.4%  (tolerance ±5.0 pts)
[headline_repro] Peak accuracy:        62.5%  (tolerance ±10.0 pts)
```

---

## 1. Prerequisites

### 1.1 Option A — Docker (recommended)

```bash
git clone https://github.com/arvindcr4/tinker-rl-lab.git
cd tinker-rl-lab
git checkout <COMMIT_HASH_FROM_ARTIFACT_MD>     # §1 of ARTIFACT.md

docker build -t tinker-rl-lab:repro .
docker run --gpus all -it \
    -e TINKER_API_KEY -e WANDB_API_KEY -e HF_TOKEN \
    -v $(pwd)/results:/workspace/tinker-rl-lab/results \
    tinker-rl-lab:repro bash
```

Build tested on: Docker 24.x, nvidia-container-toolkit ≥ 1.14, NVIDIA driver ≥ 550.

### 1.2 Option B — Local virtualenv

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional: install the project so "tinkerrl" CLI shortcuts work
pip install -e .
```

Requires CUDA 12.4+, NVIDIA driver ≥ 550, Python 3.10+.

### 1.3 Required environment variables

| Variable          | Required for                                 | Where to get it                                  |
|-------------------|----------------------------------------------|--------------------------------------------------|
| `TINKER_API_KEY`  | All Tinker experiments (GRPO, scaling)       | https://tinker-console.thinkingmachines.ai       |
| `WANDB_API_KEY`   | W&B logging (optional; set `WANDB_MODE=offline` to skip) | https://wandb.ai/authorize             |
| `HF_TOKEN`        | HuggingFace model / dataset download         | https://huggingface.co/settings/tokens           |
| `MODAL_TOKEN_ID`  | Re-running the Modal H100 PPO baselines only | https://modal.com/settings/tokens                |
| `MODAL_TOKEN_SECRET` | "                                         | "                                                |

A template lives in [`.env.example`](./.env.example).

---

## 2. The 10-minute smoke test  (reviewer entry-point)

```bash
bash scripts/smoke_test.sh                   # ~2 min — fully offline
TINKER_API_KEY=... bash scripts/smoke_test.sh # ~8 min — live Tinker wire-protocol
```

What it verifies — see the header of
[`scripts/smoke_test.sh`](./scripts/smoke_test.sh) for the full list:

1. Core library imports and versions
2. `utils.seed.set_global_seed(42)` is deterministic across Python/NumPy/PyTorch
3. `pytest tests/` green (unit tests)
4. GSM8K dataset loads and the answer-extraction regex works
5. GRPO reward + advantage normalization round-trip
6. File-surface check (all required artifact files present + parseable)
7. [optional] 3-step GRPO on `Qwen/Qwen3.5-4B` against live Tinker (wire-protocol only)

Target wall-clock: **< 10 min**. If it exceeds that, the script prints a warning
but still exits 0 so reviewers see the failure mode.

---

## 3. Headline result — GRPO Qwen3-8B GSM8K (paper Table 2)

### 3.1 Seeds

The canonical multi-seed set is **`{42, 123, 456, 789, 1024}`** (all experiments
table; see `ARTIFACT.md §4.3`). The headline single-seed run uses `seed=42`.

### 3.2 Command

```bash
for SEED in 42 123 456 789 1024; do
    python grpo_gsm8k_base.py \
        --model Qwen/Qwen3-8B \
        --seed  $SEED \
        --rank  32 \
        --steps 30 \
        --group 8 \
        --batch 2 \
        --lr    1e-5 \
        --tag   "gsm8k_qwen3_8b_s${SEED}" \
        2>&1 | tee results/gsm8k_qwen3_8b_s${SEED}.log
done
```

### 3.3 Expected per-seed output

The last line of each log is of the form:

```
[gsm8k_qwen3_8b_sSEED] Last-10 avg accuracy: XX.X%
[gsm8k_qwen3_8b_sSEED] Peak accuracy:        YY.Y%
```

| seed | last-10 (paper)* | peak (paper)* | W&B run (reference) |
|------|------------------|---------------|---------------------|
| 42   | 34.4 %           | 62.5 %        | scale_gsm8k_qwen3-8b (`vv6uu72m`) |
| 123  | 39.4 %           | 68.8 %        | grpo_qwen3-8b_gsm8k_seed123 (`c1zca2qb`) |
| 456  | 29.4 %           | 100.0 %       | grpo_qwen3-8b_gsm8k_seed456 (`igejxwwl`) |
| 789  | 28.8 %           | 68.8 %        | grpo_qwen3-8b_gsm8k_seed789 (`28bzhlsa`) |
| 1024 | —                | —             | (reviewer-only rerun)              |

*Values copied from the W&B project
`arvindcr4-pes-university/tinker-rl-lab-world-class`.  The headline value
reported in the paper is **last-10 = 34.4 %, peak = 62.5 %** — the `seed=42`
row above.

### 3.4 Aggregating the 5-seed run

```bash
python utils/stats.py \
    --results-dir results/ \
    --experiment  gsm8k_qwen3_8b \
    --rliable --bootstrap-samples 10000 \
    --output analysis/gsm8k_qwen3_8b.json
```

---

## 4. Cross-framework / ablation experiments

### 4.1 Group-size ablation (paper §4.3, `main.tex:920-925`)

```bash
for G in 2 4 8 16; do
    python grpo_gsm8k_base.py \
        --model Qwen/Qwen3-8B --seed 42 --rank 32 \
        --steps 30 --group $G --batch 2 --lr 1e-5 \
        --tag "ablation_group${G}"
done
```

Expected (paper Table 2 Group-Size block):

| G   | peak  | last-10 |
|-----|-------|---------|
| 2   | 50.0% | 37.5%   |
| 4   | 75.0% | 52.1%   |
| 8   | 100%  | 84.4%   |
| 16  | 71.9% | 38.0%   |

### 4.2 Atropos launcher (5-seed, fully parameterized YAML)

```bash
# Terminal 1 — Atropos rollout server
run-api
# Terminal 2 — environment
python atropos/tinker_atropos/environments/gsm8k_tinker.py serve \
       --config atropos/configs/gsm8k_qwen_8b.yaml
# Terminal 3 — trainer (repeat per seed)
for SEED in 42 123 456 789 1024; do
    SEED=$SEED python atropos/launch_training.py \
        --config atropos/configs/gsm8k_qwen_8b.yaml --seed $SEED
done
```

### 4.3 Size ladder (Qwen 0.6B → 30B MoE)

```bash
for CFG in gsm8k_qwen_0_6b gsm8k_qwen_1_7b gsm8k_qwen_4b \
           gsm8k_qwen_8b   gsm8k_qwen_14b \
           gsm8k_qwen_30b_moe; do
    for SEED in 42 123 456 789 1024; do
        SEED=$SEED python atropos/launch_training.py \
            --config atropos/configs/${CFG}.yaml --seed $SEED
    done
done
```

### 4.4 Modal H100 PPO baselines

```bash
python scripts/modal_run_experiments.py \
    --experiment ppo_gsm8k_qwen3_8b \
    --seeds 42 123 456 789 1024
```

Requires `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

### 4.5 Cross-library RL baselines (arithmetic env)

```bash
for exp in experiments/implementations/{trl_grpo_math,sb3_ppo_math,\
cleanrl_ppo_math,tianshou_ppo_math}.py; do
    ./scripts/run_seeds.sh "python $exp" 42 123 456 789 1024
done
```

---

## 5. Figures and tables

```bash
# 1. Aggregate JSON + CSV + rliable metrics
python utils/stats.py --results-dir results/ --rliable \
       --bootstrap-samples 10000 --output analysis/

# 2. Paper figures (matplotlib, 300 dpi PDFs)
python scripts/make_paper_figures.py \
       --results analysis/ --out paper/figures/

# 3. LaTeX tables
python utils/stats.py --results-dir results/ --latex \
       --output paper/tables/
```

---

## 6. Wall-clock budget (reviewer planning)

| Step                                   | Hardware        | Wall-clock       |
|----------------------------------------|-----------------|------------------|
| `docker build` (cold)                  | any             | ~6 min           |
| `scripts/smoke_test.sh` (offline)      | CPU             | ~2 min           |
| `scripts/smoke_test.sh` (with Tinker)  | any + net       | ~8 min           |
| Headline GRPO Qwen3-8B GSM8K (1 seed)  | 1x A100-80GB    | ~5 h             |
| Full 5-seed headline                   | 1x A100-80GB    | ~25 h            |
| Group-size ablation (4 runs × 1 seed)  | 1x A100-80GB    | ~20 h            |
| Size ladder 0.6B–14B (5-seed each)     | 1–2x A100-80GB  | ~130 h           |
| Size ladder 30B MoE (3-seed)           | 4x A100-80GB    | ~96 h            |
| **Total paper reproduction**           |                 | **~446 GPU-h**   |

A reviewer wishing to only verify the headline should plan for
**~5 GPU-hours on a single A100-80GB**, or use the smoke test
(< 10 min, CPU-only) as a sanity check.

---

## 7. Data provenance

| Dataset            | Source                                | Split / rows             | Pinned revision |
|--------------------|---------------------------------------|--------------------------|-----------------|
| GSM8K              | `openai/gsm8k` @ HF                   | `main/train` (7 473)     | `740312a` (2026-03-23) |
| GSM8K (eval)       | `openai/gsm8k` @ HF                   | `main/test`  (1 319)     | `740312a`        |
| HumanEval          | `openai/openai_humaneval` @ HF        | `test` (164)             | `2025-11-27 snapshot` |
| MATH-500           | `HuggingFaceH4/MATH-500` @ HF         | `test` (500)             | `2025-09-10 snapshot` |
| NoRobots (SFT)     | `HuggingFaceH4/no_robots` @ HF        | `train_sft` (9 500)      | `2024-11-18 snapshot` |
| OpenThoughts3 (KD) | `open-thoughts/OpenThoughts3` @ HF    | `train` (subsample 20k)  | `2025-12-02 snapshot` |

Exact shas are recorded in [`ARTIFACT.md §3.4`](./ARTIFACT.md).

---

## 8. Verifying results

```bash
python utils/verify_results.py \
    --results-dir results/ \
    --expected-results paper/expected_results.json \
    --last10-tolerance 0.05 \
    --peak-tolerance   0.10
```

**Why the tolerances are not tighter.**
Even with fixed seeds and CuDNN determinism (`utils.seed.set_global_seed`),
three sources of residual non-determinism remain:

1. **Tinker server scheduling** — remote LoRA training is not bit-deterministic
   across scheduler decisions; runs replayed months later drift by 2–4 pts on
   GSM8K last-10 (confirmed via `grpo_qwen3-8b_gsm8k_s123` re-runs:
   `c1zca2qb`, `snm9mxni`, `9lleqd1g`).
2. **CUDA / cuBLAS non-determinism** for some matmul ops on H100/A100.
3. **Sampling temperature** `0.8` with top-p 0.95 — even deterministic seeds
   still drift because the sampler is in a remote VM.

The paper's Table 10 (Welch's t-test on PPO-vs-GRPO, `p=0.7605`) shows the
run-to-run variance on this setup is of the order of the reported effect.

---

## 9. Known gotchas

- **`transformers` 4.50+** removes the `Qwen3Config` alias — pin to
  `<4.50` (enforced by `requirements.txt`). If you see
  `AttributeError: Qwen3Config`, re-check the pin.
- **GSM8K load_dataset script mode** — HF deprecated `trust_remote_code` for
  the GSM8K loader in datasets 3.0; we pass `trust_remote_code=True` only
  where needed. Upgrading past `datasets<3.2` may require a `load_dataset`
  signature change.
- **W&B offline mode** — the Docker image defaults to `WANDB_MODE=offline`
  so reviewers without a W&B account can run without errors. Remove that
  env var to log online.
- **Tinker rate limits** — new API keys are rate-limited to ~2 concurrent
  training clients. The for-loop in §3.2 runs seeds serially.

---

## 10. Contact

Open an issue on https://github.com/arvindcr4/tinker-rl-lab or email the
corresponding author listed in the paper front-matter.
