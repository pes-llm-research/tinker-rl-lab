# TinkerRL NeurIPS Paper Optimization

## Objective
Elevate this research paper and codebase to world-class, Turing-Award-worthy standards. The project is a NeurIPS submission on GRPO (Group Relative Policy Optimization) for reasoning model alignment using the Tinker cloud RL platform.

## Metric
Run `bash autoresearch_score.sh` — outputs `METRIC score=N` where N is 0-100. Higher is better.
Current baseline: 78/100.

### Unified audit suite
The unified audit suite driver is `run_all_audits.py`. It runs every `*_audit.py` script in the repo root, aggregates their individual metrics, and emits a top-level `METRIC suite_issues=N` counter (sum of per-audit non-zero metrics, where 0 = clean). Each child audit still emits its own metric line, including:

- `METRIC reviewer_issues=N` (the primary reviewer-objection counter from `paper_improvement_audit.py` and `paper_plan_audit.py`). This is the headline metric the autoresearch loop optimises against — it is the rollup of all Tier-1/2/3 reviewer concerns tracked in the discovery report.
- `METRIC caveat_issues=N` (`reviewer_caveat_audit.py`, reviewer-caveat coverage across the main paper, capstone report, and supplementary appendix).
- `METRIC capstone_issues=N`, `METRIC abstract_issues=N`, `METRIC config_issues=N`, `METRIC anon_issues=N`, `METRIC package_issues=N`, `METRIC workflow_issues=N`, `METRIC blind_package_issues=N`, `METRIC export_issues=N`, `METRIC export_guard_issues=N`, `METRIC claim_issues=N`, `METRIC sync_issues=N`, `METRIC readiness_issues=N`, `METRIC strength_issues=N`.

Any change to the paper or report is required to keep `suite_issues=0` and in particular `reviewer_issues=0`; autoresearch sessions should treat a non-zero `suite_issues` or `reviewer_issues` as a regression.

## Key Improvement Areas
1. **LaTeX quality** (currently 10/20) — Fix all warnings. Remove undefined references, fix overfull hboxes, ensure all figures/tables are referenced. Clean compilation = 20 points.
2. **Experiment results integration** (currently low) — There are 13+ completed experiments in `/tmp/campaign_v2_fixed.log` and 14 recovered results in `/home/user/workspace/elevation_outputs/campaign_recovered_results.json` that need to be added to `experiments/master_results.json`.
3. **Paper polish** — Improve abstract clarity, ensure consistent notation, add missing figure captions, ensure proper cross-references.
4. **Code documentation** — Add docstrings to key experiment scripts.

## Files in Scope
- `paper/main.tex` — The NeurIPS paper (2,877 lines)
- `paper/references.bib` — Bibliography
- `experiments/master_results.json` — Experiment results database
- `experiments/tinker-runs/*.py` — Experiment scripts
- `reports/final/capstone_final_report.md` — Companion report
- `autoresearch_score.sh` — Benchmark script

## What's Been Tried
- Nothing yet — this is the first autoresearch session

## Dead Ends
- None yet

## Key Wins
- None yet
