# experiments/CHANGELOG.md

Row-count delta log for `experiments/master_results.{json,csv}`, tracking
which task contributed which experiments.

## v3.0 — 2026-04-19 (task-13 integrator)

Regenerated `master_results.json` (schema v3.0) and `master_results.csv`
from every `experiments/**/results/*.json` file present after the tasks
1–12 consolidation.

Contributions by task (best-effort attribution from group/source columns):

| Task | Row count | Provenance |
|-----:|----------:|:-----------|
| Task 1  | 0 (missing branch; blocker issue filed) | — |
| Task 2  | 0 (missing branch; blocker issue filed) | — |
| Task 3  | 0 (missing branch; blocker issue filed) | — |
| Task 4  | **4 rows** (Modal framework-gap runs: `framework_trl_grpo`, `framework_verl_grpo`, `framework_openrlhf_grpo`, `framework_tinker_grpo_reference`) | `experiments/results/framework_comparison.json` |
| Task 5  | 0 (related-work is paper-only) | — |
| Task 6  | 0 (statistical rigor renders existing rows; adds no new experiments) | `experiments/statistical_analysis.json`, `experiments/stat_rigor_tables.json` |
| Task 7  | 0 (figures-only; consumes rows from other tasks) | `paper/figures/v2/` |
| Task 7b | 0 (paper-only figure swap) | `paper/main.tex` |
| Task 8  | 0 (repro artifact references rows from other tasks) | `Dockerfile`, `scripts/smoke_test.sh` |
| Task 9  | 0 (checklist pass; no new experiments) | `NEURIPS_CHECKLIST_FINAL.md` |
| Task 10 | 0 (ethics/limitations; no new experiments) | `paper/ethics_statement.tex` |
| Task 11 | 0 (anonymization pipeline only) | `blind_review/` |
| Task 12 | 0 (missing branch; blocker issue filed) | — |
| Legacy  | **58 rows** (pre-task-1 tinker + modal runs already on main prior to the task 1–12 campaign) | `experiments/tinker-runs/results/*.json`, `experiments/modal/`, `experiments/all_results_consolidated.json` |

**Total rows in v3:** 62 (was 79 in v2.0).

## Dead / duplicate rows archived

The v2 master contained 17 low-value duplicate rows that the integrator
removed from the live corpus. They are preserved for audit in
`experiments/_archive/removed_duplicates_2026-04-19.json`.

Breakdown of what was archived:

* **10× unnamed Qwen2.5-0.5B rows** from the ancient `collab-results/` dump
  (no experiment_id, no reward_trace, only peak/last10). These were the
  "ghost rows" called out in the task 13 spec.
* **7× exact-name duplicates** where the same experiment_id appeared twice
  in master_results.json (one with a partial trace, one with the full
  trace). The first occurrence was kept.

## Schema

Every live row guarantees the 12 required fields specified by the task-13
brief: `name, model, seed, group_size, lr, steps, peak_reward, last10_avg,
first5_avg, reward_trace, wandb_url, status`, plus 6 optional context
fields (`group, task, platform, finding, hf_checkpoint_url, zero_reward_pct`).

Empty-string sentinels are used for fields that are genuinely unknown for a
given experiment (e.g. `group_size` and `lr` for PPO runs that do not use
GRPO groups or use policy-default learning rates).
