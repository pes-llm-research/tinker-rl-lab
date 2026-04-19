# FINAL_HANDOFF — Task 13 Integrator (NeurIPS 2026 submission)

**Date:** 2026-04-19
**Branch:** `task-13-integrator` (PR into `main`)
**Tag:** `v3.0-neurips-submission` (pushed to both remotes)
**Status:** ready for submission pending a `main` merge.

This document is the handover note from the integrator to the authors
and to the post-rebuttal revision team. It consolidates what was done,
what is left, and exactly what to ship.

---

## 1. What to ship

Attach to OpenReview (Datasets & Benchmarks, blind review):

1. `submission/neurips2026_tinker_rl_lab.zip` (39 MB)
   SHA-256 `4d432d2214a7110980f3df623607d0856295ad9e8149d4652d70e689a0a33f29`

Bundle contents (from `submission/contents/MANIFEST.md`):

| File | SHA-256 |
|---|---|
| `paper_anon.pdf` (51 pages) | `c5ab440c9d3795b9c994135d75dcdb1b663b2a831856d9dfbaea864e3b254f06` |
| `paper.pdf` (52 pages, AC-only) | `15b71c30762400ee9b1ca042f231437494cfe64b34321de345c8ba363a8cffa7` |
| `code.tar.gz` (27 MB anonymised) | `5082ce24906b60f3c878b986e1fc36639524d30606bad4a998cc606768b1c976` |
| `ethics_statement.pdf` | `0204147bcfc9cad6d0110e923fbe365308734482a0a3ba82f8cd3a61dc3bb67b` |
| `data_statement.md` | `8493dd928d8ff349b905729b1c47f3894252ad959c9f9069f5ebfbac747f9c90` |
| `REVIEWER_README.md` | `3553adadcc314a40de5c5a2496fddf15485e46ea844162b77358ebdca7317ffe` |
| `MANIFEST.md` | (self-describing) |

The zip is not tracked in git; it is rebuilt deterministically by

```bash
cd /path/to/tinker-rl-lab
cd submission/contents && zip -q ../neurips2026_tinker_rl_lab.zip *.pdf *.tar.gz *.md
```

---

## 2. Verification checklist

Before hitting Submit in OpenReview, run these in order from a clean
clone of `main` at `v3.0-neurips-submission`:

```bash
git clone https://github.com/pes-llm-research/tinker-rl-lab.git
cd tinker-rl-lab
git checkout v3.0-neurips-submission

# 1. Paper builds cleanly (0/0/0/0).
cd paper && latexmk -pdf -interaction=nonstopmode main.tex && cd ..
grep -cE "^!" paper/main.log                              # -> 0
grep -cE "LaTeX Warning|Package.*Warning" paper/main.log  # -> 0
grep -cE "^Overfull|^Underfull" paper/main.log            # -> 0
grep -c  "undefined"                   paper/main.log     # -> 0

# 2. Anonymised paper builds cleanly.
cd paper && latexmk -pdf -interaction=nonstopmode main_anon.tex && cd ..

# 3. No PII in anon PDF.
pdftotext paper/main_anon.pdf - | grep -iE \
  "arvindcr4|sandhya jeya|madhu kumara|mohammad rafi|\
dhruva n murthy|arumugam k|anwesh|darapaneni|narayana|\
pes[- ]?u|pes-llm|balasandhya|madhu2133|MohammadRafiML|\
dhruvanmurthy|madhukumara1993|pes a100|by collaborator" \
  | wc -l      # must be 0

# 4. Integration audit.
python3 scripts/integration_audit.py     # 7/7 pass

# 5. Reproducibility — headline claim check (±2 pp).
python3 reproducibility/check_qwen3_8b_claim.py  # PPO + GRPO both pass
```

---

## 3. What was done in task 13

Six commits on `task-13-integrator`:

| # | SHA | Step | Description |
|---|---|---|---|
| 1 | `369f6a7` | 2+3 | Paper assembles cleanly (52 pages, figures/v2 only). |
| 2 | `bc9d423` | 2 | Consolidated `experiments/master_results.{json,csv}`, archived superseded rows, wrote `experiments/CHANGELOG.md`. |
| 3 | `2fc2a44` | 4 | `scripts/integration_audit.py` — 7 checks, all pass. |
| 4 | `b784b87` | 5 | `reproducibility/check_qwen3_8b_claim.py` + smoke test (±2 pp on headline). Data-provenance bug found & fixed in `master_results.json`: `ppo_qwen3-8b` replaced with canonical `ppo_gsm8k_Qwen3-8B_s42` trace. |
| 5 | `1fc9858` | 5 | Added offline smoke-test log (force-add past `*.log` ignore). |
| 6 | `b716473` | 6 | Rebuilt blind-review bundle from current `main.tex`. Anonymiser now processes `\input`'d sections. |
| 7 | `5b6bc40` | 7 | Root `CHANGELOG.md` for v3.0. |
| 8 | `bec6246` | 8 | Submission metadata (MANIFEST, REVIEWER_README, data_statement). |

Release engineering (step 7):

- Annotated tag `v3.0-neurips-submission` on the tip commit.
- GitHub releases cut on both remotes:
  - https://github.com/pes-llm-research/tinker-rl-lab/releases/tag/v3.0-neurips-submission
  - https://github.com/arvindcr4/tinker-rl-lab/releases/tag/v3.0-neurips-submission
- Release notes reference in-repo artefacts by path + SHA-256.

Merges on the main remote before task 13 started:

- PR #7 (task 4 — framework gap)
- PR #2 (task 7 — results & tables)
- PR #8 (task 7b — v2-figures)
- PR #9 (task 9 — NeurIPS checklist final)
- PR #10 (task 11 — anonymisation) opened & merged by integrator.

---

## 4. Known issues deferred to post-rebuttal

Integrator-blocker issues filed on the `arvindcr4/tinker-rl-lab` mirror
(because pes-llm-research has issues disabled):

| Issue | Task | Summary |
|---|---|---|
| [#4](https://github.com/arvindcr4/tinker-rl-lab/issues/4) | Task 1 | Dataset card incomplete (HF model-card template not filled for all runs). |
| [#5](https://github.com/arvindcr4/tinker-rl-lab/issues/5) | Task 2 | Scale-law extended sweep data not merged (pilot numbers remain in paper with caveat). |
| [#6](https://github.com/arvindcr4/tinker-rl-lab/issues/6) | Task 3 | ACM submission track deferred (paper ships in NeurIPS style only). |
| [#7](https://github.com/arvindcr4/tinker-rl-lab/issues/7) | Task 12 | Held-out evaluation on GSM-1k not merged; referenced only as future work. |

None of these block the blind-review submission. The paper's claims,
tables and figures are all produced by tasks 4, 7, 7b, 9, 11 that did
merge.

One paper-level caveat is worth re-reading during final review:

- Abstract and Section 4.4 cite Qwen3-8B GRPO 34.4 % vs PPO 22.5 %. The
  underlying `master_results.json` row for PPO was wrong (stub) until
  this integrator cut — it has since been replaced by the canonical
  trace from `modal_parallel_results.json` and the numbers still
  validate within ±2 pp. See `experiments/CHANGELOG.md` and
  `experiments/_archive/ppo_qwen3-8b_superseded_2026-04-19.json`.

---

## 5. How to open the PR

```bash
gh pr create \
  --repo pes-llm-research/tinker-rl-lab \
  --base main \
  --head task-13-integrator \
  --title "integrator: task 13 final submission package" \
  --body-file FINAL_HANDOFF.md
```

Merge strategy: **merge commit** (not squash) so each step's commit is
preserved in `main` history. Merge right after NeurIPS Submit Success
page is saved.

---

## 6. Post-submission actions

1. Update README.md front matter to say "NeurIPS 2026 D&B submission
   (under review)".
2. Keep `task-13-integrator` branch alive until after rebuttal.
3. Open a task 14 for post-rebuttal edits; re-run
   `scripts/integration_audit.py` before every push.
4. The W&B runs used to build `master_results.json` live at
   `arvindcr4-pes-university/tinker-rl-lab-world-class` (not
   anonymised; public on the mirror only). Do not enable wandb sync
   for the anonymous fork until after camera-ready.
