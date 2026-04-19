# Artifact Description — TinkerRL-Bench

## A Unified Benchmark for RL Post-Training of Language Models

This document follows the [ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
guidelines and the [NeurIPS 2026 reproducibility checklist](https://nips.cc/public/guides/PaperChecklist).
We target three ACM badges:

- **Artifacts Available** — permanently archived on GitHub + Hugging Face Hub (DOI pending via Zenodo).
- **Artifacts Evaluated — Functional** — documented, consistent, complete, and exercisable in < 10 min via
  [`scripts/smoke_test.sh`](./scripts/smoke_test.sh).
- **Artifacts Evaluated — Reusable** — modular layout, pinned deps, extensible configs.

---

## 1. Exact provenance  (required by NeurIPS checklist §8 and ACM §2)

| Field              | Value |
|--------------------|-------|
| **Canonical repo** | https://github.com/pes-llm-research/tinker-rl-lab |
| **Mirror repo**    | https://github.com/arvindcr4/tinker-rl-lab |
| **Tag**            | `v1.0.0-neurips-2026` *(will be pushed on camera-ready)* |
| **Head commit (this artifact)** | `9019b5c946f17ffe4761b5f4eac13b86d1e51cc0`  (short: `9019b5c`) |
| **Head commit date** | `2026-04-19 12:53:07 +0000` |
| **PR introducing this artifact** | `task-8-repro-artifact` → `main` |
| **Submitted paper** | [`paper/main.tex`](./paper/main.tex) |
| **Capstone report** | [`reports/capstone_final_report.md`](./reports/capstone_final_report.md) |
| **W&B project**    | `arvindcr4-pes-university/tinker-rl-lab-world-class` (153 runs, 23.9 h client-side wall-clock) |
| **License**        | Apache-2.0 (see [`LICENSE`](./LICENSE)) |
| **DOI**            | _Pending — will be minted on Zenodo upon camera-ready_ |

To verify you have the right commit:

```bash
git rev-parse HEAD   # should print 9019b5c946f17ffe4761b5f4eac13b86d1e51cc0
```

The Docker image records the same commit in `/workspace/tinker-rl-lab/.build_info`
and as an OCI label (`org.opencontainers.image.revision`).

---

## 2. Repository layout

```
tinker-rl-lab/
├── Dockerfile                        # Pinned CUDA 12.4 + Ubuntu 22.04 env
├── REPRODUCE.md                      # Step-by-step reproduction (this pairs with ARTIFACT.md)
├── ARTIFACT.md                       # This file
├── ACM_CHECKLIST.md / NEURIPS_CHECKLIST.md
├── COMPUTE.md                        # Per-experiment GPU budget (paper §G)
├── LIMITATIONS_AND_IMPACT.md
├── requirements.txt                  # Pinned Python deps
├── pyproject.toml                    # Packaging + optional extras
├── scripts/
│   ├── smoke_test.sh                 # <10 min reviewer entry-point  ← NEW (Task 8)
│   ├── run_seeds.sh                  # Multi-seed driver
│   ├── make_paper_figures.py         # Regenerate paper figures
│   ├── modal_run_experiments.py      # Modal H100 PPO baselines
│   └── anonymize.sh                  # Double-blind export
├── utils/
│   ├── seed.py                       # set_global_seed(): Python/NumPy/PyTorch/CUDA
│   └── stats.py                      # rliable + bootstrap CIs
├── grpo_gsm8k_base.py                # Headline GRPO on Qwen3-8B GSM8K
├── grpo_exp_{a..d}.py                # Seeded ablations (A/B/C/D)
├── grpo_tooluse_tinker.py            # Tool-use task
├── experiments/
│   ├── implementations/              # 13 trainer scripts (TRL, SB3, CleanRL, Tianshou, d3rlpy, PufferLib, rl-games)
│   ├── 10x_structural_ceiling/       # 31 block-A..J YAML configs — scaling ceiling study
│   ├── modal/                        # Modal H100 runners
│   └── notebooks/                    # Jupyter / Colab versions
├── atropos/                          # Atropos integration + 34 YAML configs
├── paper/                            # main.tex + figures + tables
├── reports/capstone_final_report.md  # Capstone report
└── tests/                            # pytest smoke suite (imported by scripts/smoke_test.sh)
```

---

## 3. Environment & dependencies

### 3.1 Pinned environment (Docker, recommended)

The `Dockerfile` bakes everything needed to reproduce the paper:

- **Base image**: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`
- **Python**: 3.10 (from Ubuntu 22.04 apt)
- **Init**: `tini` as PID 1 for clean signal handling
- **Determinism env**: `PYTHONHASHSEED=42`, `SEED=42`, `TOKENIZERS_PARALLELISM=false`
- **Provenance labels**: `org.opencontainers.image.revision=<git sha>`, baked at build time via `--build-arg GIT_COMMIT=...`
- **Self-check**: Dockerfile RUN step imports torch/transformers/trl/numpy/scipy and fails the build if any version mismatches.

Image digest is printed on every build and is also reproducible via
`docker buildx build --platform linux/amd64 --sbom=true`.

### 3.2 Hardware requirements

| Resource    | Minimum             | Recommended (paper config) |
|-------------|---------------------|----------------------------|
| GPU         | NVIDIA A100 40GB (inference only) | NVIDIA A100 80GB (Qwen3-8B training) |
| VRAM        | 24 GB (≤1B models)  | 80 GB (Qwen3-8B LoRA r=32) |
| System RAM  | 32 GB               | 64 GB                      |
| Disk        | 50 GB               | 200 GB (checkpoints + HF cache) |
| CUDA driver | 550+                | 555+                       |
| Host OS     | Linux x86_64        | Ubuntu 22.04               |

Tinker API runs train remotely on Thinking Machines' infrastructure, so the
**only hard local GPU requirement** is for Modal baselines and the size-ladder
inference (§6.4 of `REPRODUCE.md`).

### 3.3 Pinned library versions  (excerpt — full list in `requirements.txt`)

| Package          | Pin                 |
|------------------|---------------------|
| torch            | `>=2.3.0,<2.5.0`    |
| numpy            | `>=1.26.0,<2.0.0`   |
| transformers     | `>=4.46.0,<4.50.0`  |
| trl              | `>=1.0.0,<1.2.0`    |
| datasets         | `>=3.0.0,<3.2.0`    |
| accelerate       | `>=1.0.0,<1.2.0`    |
| peft             | `>=0.13.0,<0.15.0`  |
| stable-baselines3| `>=2.3.0,<2.5.0`    |
| tianshou         | `>=1.1.0,<1.2.0`    |
| d3rlpy           | `>=2.6.0,<2.8.0`    |
| tinker           | `>=0.5.0`           |
| tinker-cookbook  | `>=0.2.0`           |
| atroposlib       | `>=0.1.0`           |
| rliable          | `>=1.1.0,<1.3.0`    |
| wandb            | `>=0.18.0,<0.20.0`  |

### 3.4 Dataset + model revisions (pinned)

All datasets are consumed via `datasets.load_dataset(..., revision=<sha>)`
when a deterministic run is required; the default loaders use the *latest*
revision at training time. Exact shas as of artifact submission:

| Asset                        | Hub ID                          | Revision SHA (first 10)    | Dated              |
|------------------------------|---------------------------------|----------------------------|--------------------|
| GSM8K (train/test)           | `openai/gsm8k`                  | `740312add8`               | 2026-03-23         |
| HumanEval                    | `openai/openai_humaneval`       | `7dce6050a7`               | 2024-01-04         |
| MATH-500                     | `HuggingFaceH4/MATH-500`        | `6e4ed1a2a7`               | 2025-12-15         |
| NoRobots (chat SFT)          | `HuggingFaceH4/no_robots`       | `e6f9a4ac5c`               | 2024-04-18         |
| OpenThoughts3 (distillation) | `open-thoughts/OpenThoughts3`   | snapshot copy at `data/openthoughts3_sub20k.parquet` | 2025-12-02 |
| Qwen3-8B                     | `Qwen/Qwen3-8B`                 | `b968826d9c`               | 2025-07-26         |
| Qwen3-8B-Base                | `Qwen/Qwen3-8B-Base`            | `49e3418fbb`               | 2025-05-21         |
| Qwen3.5-4B                   | `Qwen/Qwen3.5-4B`               | `851bf6e806`               | 2026-03-02         |
| Llama-3.1-8B-Instruct        | `meta-llama/Llama-3.1-8B-Instruct` | `0e9e39f249`            | 2024-09-25         |

`scripts/smoke_test.sh` step 4 loads the first 8 GSM8K rows to verify the
answer-extraction regex works against whatever revision the reviewer's HF
cache has.

---

## 4. Headline result (what reviewers must verify)

### 4.1 The single number

> **GRPO on Qwen3-8B, GSM8K, 30 steps, LoRA rank 32, group size 8, seed 42**
> Peak accuracy **62.5 %**, last-10-step average **34.4 %**.
> Reported in `paper/main.tex` line 897 (Table 2).

### 4.2 Exact command (from `REPRODUCE.md §3.2`)

```bash
python grpo_gsm8k_base.py \
    --model Qwen/Qwen3-8B \
    --seed  42 \
    --rank  32 \
    --steps 30 \
    --group 8  \
    --batch 2 \
    --lr    1e-5 \
    --tag   headline_repro
```

### 4.3 Canonical seed list

`{42, 123, 456, 789, 1024}` — used for every statistical test in the paper.
Ablation-only seeds (for 10× structural-ceiling blocks A/B/D/E):
`{42001, 42002, 42003, 42004, 42005}` (see `experiments/10x_structural_ceiling/configs/`).

### 4.4 W&B pointer runs (each row = one seed, same hyper-params as §4.2)

| Seed | W&B run ID | Display name                    | Peak  | Last-10 | Wall-clock |
|------|-----------|---------------------------------|-------|---------|------------|
| 42   | `vv6uu72m` | scale_gsm8k_qwen3-8b            | 62.5% | 34.4%   | ~5.0 h *    |
| 123  | `c1zca2qb` | grpo_qwen3-8b_gsm8k_seed123     | 68.8% | 39.4%   | 976 s       |
| 456  | `igejxwwl` | grpo_qwen3-8b_gsm8k_seed456     | 100%  | 29.4%   | 1019 s      |
| 789  | `28bzhlsa` | grpo_qwen3-8b_gsm8k_seed789     | 68.8% | 28.8%   | 1184 s      |

\* W&B client runtime is ~20 min; the bulk (~4.5 h) runs on the Tinker backend.

---

## 5. Wall-clock and compute cost

### 5.1 Per-experiment budget (reported in paper §G, `COMPUTE.md`)

| Experiment block                               | GPU                | GPUs | Time/seed | Seeds | GPU-hrs |
|------------------------------------------------|--------------------|------|-----------|-------|---------|
| Math RL (Arithmetic) — cross-library           | A100-40GB          | 1    | ~30 min   | 5     | 2.5     |
| Math RL (GSM8K) — TRL/CleanRL baselines        | A100-80GB          | 1    | ~4 h      | 5     | 20      |
| Chat SFT (NoRobots, Llama-3.2-1B)              | A100-40GB          | 1    | ~2 h      | 5     | 10      |
| DPO Shorter (Qwen3-0.6B)                       | A100-40GB          | 1    | ~1 h      | 5     | 5       |
| Distillation (off-policy, Llama-3.2-1B)        | A100-40GB          | 1    | ~1.5 h    | 5     | 7.5     |
| Distillation (on-policy, Llama-3.2-1B)         | A100-80GB          | 1    | ~3 h      | 5     | 15      |
| Atropos GSM8K — Llama-3.2-3B                   | A100-80GB          | 1    | ~3 h      | 5     | 15      |
| Atropos GSM8K — Llama-3.1-8B                   | A100-80GB          | 1    | ~5 h      | 5     | 25      |
| Atropos GSM8K — Qwen3-4B                       | A100-80GB          | 1    | ~3 h      | 5     | 15      |
| **Atropos GSM8K — Qwen3-8B (headline)**        | **A100-80GB**      | **1**| **~5 h**  | **5** | **25**  |
| Atropos GSM8K — Qwen3-14B                      | A100-80GB          | 2    | ~6 h      | 5     | 60      |
| Atropos GSM8K — Qwen3-30B-A3B (MoE)            | A100-80GB          | 4    | ~8 h      | 3     | 96      |
| **Reported experiments total**                 |                    |      |           |       | **~296**|
| Preliminary / failed experiments               |                    |      |           |       | ~100    |
| Hyperparameter sweeps (LR, group size, rank)   |                    |      |           |       | ~50     |
| **Total project compute**                      |                    |      |           |       | **~446 A100 GPU-h** |

Client-side wall-clock measured by W&B across 153 runs: **23.9 h**
(most Tinker compute happens server-side and is not captured by `_runtime`).

### 5.2 Cost estimate

| Provider / line item          | Rate                              | Hours | Cost (USD) |
|-------------------------------|-----------------------------------|-------|------------|
| Tinker API (GRPO + scaling)   | see [rate card](https://tinker-console.thinkingmachines.ai/rate-card) | ~200 h | credits     |
| GCP A100-40GB `a2-highgpu-1g` | $3.67 / GPU-h                     | ~100 h | ~$367      |
| GCP A100-80GB `a2-highgpu-2g` | $5.00 / GPU-h                     | ~146 h | ~$730      |
| Modal H100 PPO baselines      | $6.90 / GPU-h                     | ~6 h  | ~$41       |
| **Total estimated cost**      |                                   |       | **~$1,138 + Tinker credits** |

Carbon footprint (Strubell-style, 400 W TDP × 1.1 PUE × 0.39 kg CO₂/kWh) ≈ **76 kg CO₂**.

---

## 6. Determinism & tolerances

### 6.1 Seeding surface

`utils.seed.set_global_seed(seed, deterministic_cudnn=True)` sets:

- `PYTHONHASHSEED` (via env at process start)
- `random.seed(seed)`
- `numpy.random.seed(seed)`
- `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)`
- `torch.backends.cudnn.deterministic = True`, `.benchmark = False`
- `transformers.set_seed(seed)` (when importable)

### 6.2 Residual non-determinism (why we accept ±5 pts on last-10)

1. **Tinker remote scheduler** — training clients run on a shared backend whose
   scheduling and tensor-parallel assignment are not bit-identical across calls.
   Measured drift on `grpo_qwen3-8b_gsm8k_seed123` (re-runs `c1zca2qb`/`snm9mxni`/
   `9lleqd1g`): **σ ≈ 2.1 pts** on last-10.
2. **CUDA / cuBLAS non-determinism** for some fused matmuls on H100s.
3. **Sampling stochasticity** at T=0.8, top-p=0.95 — seeded but drawn from
   remote samplers that re-initialize on each call.

Tolerance policy (enforced by `utils/verify_results.py`):

- `--last10-tolerance 0.05` (±5 pts absolute)
- `--peak-tolerance   0.10` (±10 pts absolute)

The paper's Welch t-test (Table 10) shows run-to-run variance on this setup is
comparable to the reported PPO-vs-GRPO effect (p = 0.7605), which motivates the
chosen tolerance.

---

## 7. Reviewer flow (matches ACM evaluation criteria)

| ACM criterion         | Evidence                                                              |
|-----------------------|------------------------------------------------------------------------|
| **Documented**        | `README.md`, `REPRODUCE.md`, `ARTIFACT.md`, inline docstrings, 2 checklists |
| **Consistent**        | W&B runs (§4.4) reproduce the paper table; verified via `utils/verify_results.py` |
| **Complete**          | 13 trainer scripts + 34 Atropos configs + 31 scaling configs + stats tools |
| **Exercisable**       | `scripts/smoke_test.sh` (< 10 min, 7 checks); `scripts/run_seeds.sh`; `python grpo_gsm8k_base.py` |
| **Verification**      | `utils/stats.py --rliable` computes IQM, bootstrap CIs, performance profiles |
| **Well-structured**   | Modular layout: `experiments/`, `utils/`, `scripts/`, `atropos/` (§2) |
| **Docs beyond min.**  | `NEURIPS_CHECKLIST.md`, `ACM_CHECKLIST.md`, `LIMITATIONS_AND_IMPACT.md`, `COMPUTE.md` |
| **Standards**         | ACM v1.1 badges, NeurIPS 2026 checklist, HF model-card spec            |
| **Extensible**        | New RL libraries drop into `experiments/implementations/<name>.py`; new tasks → new YAML under `atropos/configs/` |

---

## 8. Archival

| Channel              | Purpose                                     | Persistence   |
|----------------------|---------------------------------------------|---------------|
| GitHub (canonical)   | Source, scripts, docs                       | Permanent (public repo) |
| GitHub (mirror)      | Back-up of source                           | Permanent (public repo) |
| Hugging Face Hub     | Trained LoRA adapters + model cards         | Permanent (HF storage) |
| Weights & Biases     | Run logs (public project)                   | Indefinite (W&B retention) |
| Zenodo (post-CR)     | Snapshot DOI tied to `v1.0.0-neurips-2026`  | Permanent (DOI) |

---

## 9. Contact

File an issue on https://github.com/arvindcr4/tinker-rl-lab or email the corresponding author listed in `paper/main.tex`.

---

_This artifact description follows the
[ACM Artifact Review and Badging v1.1](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
and the NeurIPS 2026 reproducibility checklist._
