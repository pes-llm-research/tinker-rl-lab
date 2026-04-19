# Changelog

All notable changes to Tinker RL Lab are recorded here. The project uses
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions and
[semantic versioning](https://semver.org/) for its public research artefacts.

Task-level changelogs live in their natural homes (for example,
`paper/CHANGELOG.md` for the paper and `experiments/CHANGELOG.md` for the
results CSV/JSON). This file rolls up the repository-wide release history.

---

## [v3.0-neurips-submission] — 2026-04-19

**NeurIPS 2026 Datasets & Benchmarks camera-ready / blind submission.**

This release is the integrator cut of task 13. Every task 1–12 artefact
referenced by the paper has been verified against `scripts/integration_audit.py`
(7/7 checks pass) and exercised by `reproducibility/check_qwen3_8b_claim.py`
and `reproducibility/smoke_test_2026-04-19.log` (±2 pp of the Qwen3-8B
GRPO claim).

### Added
- `scripts/integration_audit.py` — 7 integration checks covering abstract
  claims, figure→result wiring, checklist↔paper consistency, bib hygiene,
  ethics-statement numerics, compute accounting, and author-list parity.
  Writes `integration_audit.json` for CI.
- `reproducibility/check_qwen3_8b_claim.py` — ±2 pp tolerance check for
  the headline Qwen3-8B GRPO-vs-PPO result; both configurations pass
  (PPO Δ = 0.0 pp, GRPO Δ = 0.025 pp).
- `reproducibility/smoke_test_2026-04-19.log` — offline 6/6 test subset,
  25 s wall-clock on CPU.
- `INTEGRATION_LOG.md` — per-commit integrator record, test-merge matrix,
  conflict resolutions, and deferred work.
- `CITATION.cff` (CFF 1.2.0) listing all eight authors.
- `CHANGELOG.md` (this file).
- `paper/FIGURES.tex` — single source of truth for every figure in the
  paper, pointing at `figures/v2/*` exclusively.
- Blind-review bundle: `blind_review/SUBMISSION_MANIFEST.md`,
  `blind_review/main_anon.pdf`, `blind_review/main_anon.tex`,
  regenerated `blind_review/tinker-rl-lab-anon.tar.gz`.

### Changed
- `paper/main.tex` now `\input`s every section module explicitly (no
  silent orphans), uses `figures/v2/` exclusively, and compiles cleanly
  (`latexmk -pdf -interaction=nonstopmode main.tex`) with 0 errors,
  0 warnings, 0 overfull/underfull boxes >1 pt, and 0 undefined
  references (52 pages, 6.7 MB).
- `experiments/master_results.json` / `master_results.csv` replaced the
  prior `ppo_qwen3-8b` stub with the canonical `ppo_gsm8k_Qwen3-8B_s42`
  trace from `modal_parallel_results.json` (peak 0.75, last10 0.225).
  The superseded row is archived at
  `experiments/_archive/ppo_qwen3-8b_superseded_2026-04-19.json`.
- `experiments/CHANGELOG.md` records the amendment.
- `paper/references.bib` removed 3 never-cited entries
  (`wolfe2026grpotricks`, `raschka2025statellms`,
  `li2025reward_shaping`); restored 2 that are cited
  (`hatamizadeh2026igrpo`, `zhang2024gsm1k`). 157 entries total.
- `blind_review/anonymize_paper.py` now processes `\input`'d section
  files and has substitution rules for residual PII discovered in
  task 9/11 (e.g. `PES A100`, `by collaborator <Name>`,
  `arvindcr4/tinker-rl-bench-*`, `arvindcr4-pes-university` wandb).
- `blind_review/anonymize_code.py` drops `CITATION.cff` from the
  anonymised tarball.
- `README.md` author list updated to the eight canonical contributors.

### Fixed
- Task 4 framework-gap table (`paper/sections/framework_gap.tex`) no
  longer cross-refs the deleted `fig:framework_gap_v1` figure.
- Task 7 results table now sources every cell from
  `master_results.csv` (no hand-edited numbers).
- Task 9 NeurIPS checklist answer for Q14 no longer mentions
  `arvindcr4-pes-university` by name.

### Security / anonymity
- Anonymised PDF (51 pages) passes `pdftotext | grep` for every team
  name, handle, institution, and private repo slug with zero hits.
- Anonymised tarball passes `_post_scan` in `anonymize_code.py` with
  zero residual identifiers after CITATION.cff exclusion.

### Integrator-blocker issues (filed on `arvindcr4/tinker-rl-lab` mirror)
- `#4` Task 1 — dataset card still incomplete
- `#5` Task 2 — scale-law data not merged
- `#6` Task 3 — acm submission track (deferred)
- `#7` Task 12 — held-out evaluation not merged

None of these block the blind-review submission; they are captured for
the post-rebuttal revision.

---

## Earlier history

Pre-v3.0 changes are tracked at task granularity inside the respective
PRs merged into `main` (tasks 4, 7, 7b, 9, 11). See `INTEGRATION_LOG.md`
for the merge sequence and test-merge matrix.
