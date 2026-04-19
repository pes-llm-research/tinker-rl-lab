# Data Statement — Tinker RL Lab (NeurIPS 2026 D&B submission)

## 1. Datasets used as-is (no redistribution)

The benchmark harness evaluates on existing public datasets; we do **not**
redistribute any of them.

| Dataset | License | Source |
|---|---|---|
| GSM8K | MIT | https://github.com/openai/grade-school-math |
| MATH | MIT | https://github.com/hendrycks/math |
| HumanEval | MIT | https://github.com/openai/human-eval |
| MBPP | Apache-2.0 | https://github.com/google-research/google-research/tree/master/mbpp |
| GSM-1k (held-out) | MIT (Scale AI) | https://scale.com/leaderboard/gsm1k |
| CountDown (tool-use) | MIT | https://github.com/Jiayi-Pan/TinyZero |

Each evaluation in `experiments/master_results.json` records the exact
dataset slug and split used. No new dataset is introduced by this paper.

## 2. Artefacts introduced by this paper

| Artefact | License | Description |
|---|---|---|
| Tinker RL Lab benchmark harness | Apache-2.0 | Code in this repo |
| LoRA adapter checkpoints (Modal runs) | Apache-2.0 | Posted to HuggingFace (anonymous mirror for review) |
| Experiment manifest (`master_results.json`) | Apache-2.0 | 79 runs × 7 frameworks × 5 model families |
| Figures & tables | CC BY 4.0 | Regenerated deterministically from `master_results.json` |

## 3. Personally identifiable information

None of the datasets contain PII. We collected no new human-subject data;
no IRB was required.

## 4. Offensive / biased content

Reward functions operate on short numeric or code-execution rewards; they
do not expose models to offensive content beyond what is already present
in the listed public datasets.

## 5. Known limitations

- No multilingual evaluation; all datasets are English.
- No non-English code generation; HumanEval/MBPP are Python-only.
- No long-form reasoning beyond 1024 tokens for GSM8K/MATH.

Limitations are discussed in full in Section "Limitations & Impact" of
the paper and in `LIMITATIONS_AND_IMPACT.md`.

## 6. Reproducibility contact

For reproducibility questions please use the OpenReview submission
thread; the anonymised code at
`https://anonymous.4open.science/r/tinker-rl-lab` contains a `REPRODUCE.md`
and a Dockerfile.
