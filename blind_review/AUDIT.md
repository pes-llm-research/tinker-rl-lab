# Task 11 — Anonymization & Blind-Review Sweep — Audit Report

Branch: `task-11-anonymization`
Target: NeurIPS 2026 blind review (deadline 2026-05-06)
Paper: `paper/main.tex` (non-anon, untouched) → `paper/main_anon.tex` (anonymised)

This document is the authoritative record of every change made for the
blind-review bundle. **The non-anon working tree was NOT modified.** All
anonymisation artefacts live under `blind_review/` and `paper/main_anon.tex`.

---

## 1. Blind-review deliverables

| Deliverable | Path | Notes |
|---|---|---|
| Anonymised paper | `paper/main_anon.tex` | 445 lines, NeurIPS anonymous-submission style |
| Anonymised code tarball | `blind_review/tinker-rl-lab-anon.tar.gz` | 22,514,454 bytes, 539 files |
| Tarball SHA-256 | `ef03e4ab3855e33d810a27c73296e90271efd1a4e39283b553d09b53744fb011` | |
| Anonymisation scripts | `blind_review/anonymize_paper.py`, `blind_review/anonymize_code.py` | Idempotent; reviewers can re-run |
| Paper change log | `blind_review/paper_changes.log` | Per-rule replacement counts |
| Code change log | `blind_review/code_changes.log` | Per-file and per-rule counts |
| Audit logs | `blind_review/audit_logs/` | Output of all `*_audit.py` scripts |

---

## 2. Audit-script results

All 17 `*_audit.py` scripts at the repo root were executed. `run_all_audits.py`
covers 13 of them; 4 extras (`autoresearch_config_audit.py`,
`paper_plan_audit.py`, `reviewer_caveat_audit.py`, `scientific_audit.py`) were
executed separately. Logs are in `blind_review/audit_logs/`.

### 2.1 Passing (10)

| Audit | Metric | Status |
|---|---|---|
| `submission_claim_audit.py` | `claim_issues=0` | pass |
| `paper_sync_audit.py` | `sync_issues=0` | pass |
| `heldout_readiness_audit.py` | `readiness_issues=0` | pass |
| `anonymization_repro_audit.py` | `anon_issues=0` | pass |
| `claim_strength_audit.py` | `strength_issues=0` | pass |
| `submission_package_audit.py` | `package_issues=0` | pass |
| `submission_workflow_audit.py` | `workflow_issues=0` | pass |
| `blind_review_package_audit.py` | `blind_package_issues=0` | pass |
| `blind_review_export_audit.py` | `export_issues=0` | pass |
| `export_guard_audit.py` | `export_guard_issues=0` | pass |

### 2.2 Failing — pre-existing, not in scope for Task 11

Task 11 scope is "Do not modify the non-anon tree." The following failures
exist in the current non-anon tree and are documented here for the record;
fixing them belongs to other tasks in the submission plan.

| Audit | Issue(s) | Resolution owner |
|---|---|---|
| `paper_improvement_audit.py` (rc=1, `reviewer_issues=1`) | `T2_conclusion_heldout`: conclusion doesn't mention need for held-out evaluation | paper-hardening task |
| `paper_plan_audit.py` (rc=1) | Same `T2_conclusion_heldout` item | paper-hardening task |
| `capstone_claim_audit.py` (rc=0, `capstone_issues=1`) | `missing_baseline_positioning` | paper-hardening task |
| `abstract_scope_audit.py` (rc=0, `abstract_issues=4`) | Missing `humaneval_subset_caveat` and `training_reward_caveat` in the abstract of both `main.tex` and `main_anon.tex` | abstract-caveats task |
| `reviewer_caveat_audit.py` (rc=0, `caveat_issues=6`) | `heldout_scope`, `tool_eval_protocol`, `codegen_subset`, `budget_and_splits`, `replication_release`, `report.threshold_overclaim` | reviewer-caveats task |
| `autoresearch_config_audit.py` (rc=0, `config_issues=3`) | `autoresearch_md_missing_suite_metric`, `autoresearch_md_missing_unified_suite_reference`, `autoresearch_md_missing_primary_reviewer_metric_context` | autoresearch-config task |

### 2.3 Environment limitation

- `scientific_audit.py` terminated with
  `FileNotFoundError: [Errno 2] No such file or directory: 'pdflatex'`. The
  sandbox does not ship a TeX Live install; the audit checks LaTeX build
  hygiene. Running on the submission runner (which has pdflatex) is expected
  to succeed; see `blind_review/audit_logs/scientific_audit.log`.

---

## 3. Paper anonymisation (`paper/main_anon.tex`)

Source: `paper/main.tex`. Produced by `blind_review/anonymize_paper.py`.

### 3.1 Structural changes

- Author block replaced with `\author{Anonymous Author(s) \\ Anonymous Affiliation \\ \texttt{anonymous@neurips.cc}}`.
- NeurIPS style flag switched from final-camera to anonymous submission
  (`\usepackage{neurips_2026}` / `\usepackage[final]{...}` → anonymous mode).
- Header `% TinkerRL Lab` project tag removed.
- Header `% Author guidance: ...` and inline `% for submission` comments removed.
- `\section*{Acknowledgments}` / `\begin{ack}...\end{ack}` contents replaced with
  `We will add acknowledgments in the camera-ready version.` stub.
- `Team Model Checkpoints` block replaced with an anonymous stub.

### 3.2 Identifier rewrites in paper body

Replacement counts from `blind_review/paper_changes.log` (non-zero only):

| Rule | Count |
|---|---|
| author block replaced with Anonymous Author(s) | 1 |
| neurips style flag switched to anonymous submission | 1 |
| header author-guidance comment removed | 1 |
| inline `for submission` comment removed | 1 |
| header `TinkerRL Lab` project tag removed | 1 |
| `github.com/arvindcr4/tinker-rl-lab` → `anonymous.4open.science` | 4 |
| HuggingFace `arvindcr4/tinker-rl-bench-*` → `anonymous/` | 6 |
| wandb project URL → anonymous mirror | 3 |
| wandb project name `tinker-rl-lab-world-class` → `tinker-rl-bench` | 2 |
| wandb project name (bare) → `tinker-rl-bench` | 1 |
| `Team Model Checkpoints` block replaced with anonymous stub | 1 |
| `\begin{ack}..\end{ack}` anonymised | 1 |
| `\section*{Acknowledgments}` anonymised | 1 |
| `PES University's LLM Research Group` redacted | 1 |

### 3.3 Verification

Post-scan against `paper/main_anon.tex` for the union of identifier tokens
(`arvindcr4|pes-llm-research|Madhu|Sandhya|Mohammad Rafi|Dhruva|Arumugam|
Anwesh|Narayana|Padhuri|Paduri|Jeyaraj|PES University|Great Learning|
Northwestern|Madhu2133|MohammadRafiML|Balasandhya|dhruvanmurthy|
tinker-rl-lab-world-class|PES2PGE`) returns zero matches.

---

## 4. Code anonymisation (`blind_review/tinker-rl-lab-anon.tar.gz`)

Source: output of `git ls-files --cached --others --exclude-standard` from
the working tree (tracked + newly-added files, respecting `.gitignore`).
Produced by `blind_review/anonymize_code.py`.

### 4.1 Substitution rules (longest-match first)

Rules use explicit lookaround `(?<![A-Za-z0-9_])…(?![A-Za-z0-9_])` instead of
`\b` so hyphenated slugs like `pes-llm-research` match correctly even when
preceded by `\n` inside Python string literals.

| Rule | Replacement |
|---|---|
| `arvindcr4-pes-university` | `anonymous-entity` |
| `tinker-rl-lab-world-class` | `tinker-rl-bench` |
| HF handles (`Madhu2133`, `MohammadRafiML`, `Balasandhya`, `dhruvanmurthy`) | `anonymous` |
| GH handles (`madhukumara1993`, `arvindcr4`) | `anonymous` |
| `pes-llm-research` | `anonymous-org` |
| `PES LLM Research Team` | `Anonymous Authors` |
| `PES LLM Research` | `Anonymous` |
| `PES University`, `Great Learning`, `Northwestern University` | `Anonymous Institution` |
| `Narayana Darapaneni`, `Anwesh Reddy Padhuri`, `Anwesh Reddy Paduri`, `Sandhya Jeyaraj`, `Madhu Kumara L`, `Mohammad Rafi`, `Dhruva N Murthy`, `Arumugam Chetty K`, `Arumugam K`, `Arvind C R` | `Anonymous` |
| First names (`Sandhya`, `Madhu`, `Arvind`, `Dhruva`, `Anwesh`, `Rafi`, `Arumugam`) | `Anonymous` |
| Lowercase slug `arumugam` in ID keys | `anonymous` |
| `PES2PGE\d{2}DS\d{3}` (student IDs) | `ANONYMIZED-ID` |
| 9 team/supervisor email addresses | `anonymous@neurips.cc` |
| `github.com/{anonymous,anonymous-org}/tinker-rl-lab` | `https://anonymous.4open.science/r/tinker-rl-lab` |

### 4.2 Totals by rule (from `blind_review/code_changes.log`)

| Rule | Hits |
|---|---|
| wandb project `tinker-rl-lab-world-class` → `tinker-rl-bench` | 39 |
| `PES University` → `Anonymous Institution` | 34 |
| GH handle `arvindcr4` → `anonymous` | 32 |
| GH org `pes-llm-research` → `anonymous-org` | 27 |
| wandb entity `arvindcr4-pes-university` → `anonymous-entity` | 24 |
| `github.com/anonymous-org/tinker-rl-lab` → `anonymous.4open.science` | 21 |
| notebook `metadata.colab` removed | 12 |
| `Mohammad Rafi` → `Anonymous` | 10 |
| First name `Madhu` → `Anonymous` | 8 |
| First name `Arumugam` → `Anonymous` | 6 |
| `Sandhya Jeyaraj` → `Anonymous` | 6 |
| `Madhu Kumara L` → `Anonymous` | 6 |
| `PES LLM Research Team` → `Anonymous Authors` | 5 |
| First name `Sandhya` → `Anonymous` | 5 |
| `PES LLM Research` → `Anonymous` | 4 |
| `Arvind C R` → `Anonymous` | 4 |
| `Arumugam Chetty K` → `Anonymous` | 4 |
| `github.com/anonymous/tinker-rl-lab` → `anonymous.4open.science` | 3 |
| First name `Arvind` → `Anonymous` | 3 |
| First name `Dhruva` → `Anonymous` | 3 |
| GH handle `madhukumara1993` → `anonymous` | 3 |
| Lowercase `arumugam` (slug/id) → `anonymous` | 3 |
| `Dhruva N Murthy` → `Anonymous` | 3 |
| `Arumugam K` → `Anonymous` | 2 |
| LICENSE copyright line (already anonymous) | 1 |
| email redacted | 1 |
| `Narayana Darapaneni` → `Anonymous` | 1 |
| `Anwesh Reddy Padhuri` → `Anonymous` | 1 |

70 files were edited in total. Notebook (`.ipynb`) metadata sections with
`authors`, `author`, `kaggle`, `colab`, or `user` keys were removed as a
defence-in-depth measure (12 hits).

### 4.3 Excluded files / directories

These paths were removed from the anonymised bundle because they either
(a) contain un-rewritable identifying content (binary office files with
embedded author metadata), or (b) duplicate the anonymised paper, or
(c) are leftover build artefacts / internal planning docs that should not
ship to reviewers:

Excluded files:

- `autoresearch-dashboard.md`
- `experiments/collab-results/LLM_Tool_Call_Finetuning.pptx`
- `experiments/collab-results/exp3_multiturn.pptx`
- `paper/acm_main.tex`
- `paper/ethics_statement.tex`
- `paper/limitations_update.tex`
- `paper/main.aux`, `paper/main.bbl`, `paper/main.blg`, `paper/main.out`, `paper/main.pdf`, `paper/main.tex`
- `paper/neurips_checklist_update.tex`
- `reports/final/CONSOLIDATED_REVIEW_IMPROVEMENTS.md`
- `reports/final/PAPER_IMPROVEMENT_PLAN.md`
- `reports/final/capstone_final_report.docx`
- `reports/final/capstone_final_report.md`
- `reports/final/grpo_agentic_llm_paper.md`
- `reports/final/grpo_agentic_llm_paper.tex`
- `scripts/anonymize.sh`
- `team-analysis.pplx.md`
- `team-links-audit.pplx.md`
- `verify_links_entities.txt`

Excluded directories:

- `blind_review/`
- `past_session_contexts/`
- `wandb/`
- `__pycache__/`

### 4.4 Included binaries (figures & data zip)

Figures under `paper/figures/`, `paper/tikz/`, `grpo_ablation_results/images/`,
and `reports/final/*.png` are copied as-is. The data archive
`GRPO_Ablation_results.zip` is included. All have been scanned with `strings`
for the identifier token set and contain no identifying strings.

### 4.5 Post-scan verification

`blind_review/anonymize_code.py` performs a final bytewise scan (reading each
file with `read_bytes()` and decoding as UTF-8 with `errors="ignore"`) over
the full anonymised tree for the identifier token union. The final run
reports: **"Post-scan: no residual identifiers detected"** (see
`blind_review/code_changes.log`).

A manual cross-check using `grep -rE` and `strings` over all text, PDF,
PNG, and ZIP files in `blind_review/tinker-rl-lab-anon/` returned zero hits.

---

## 5. Reproduction

From a fresh clone on `task-11-anonymization`:

```
python blind_review/anonymize_paper.py
python blind_review/anonymize_code.py
python run_all_audits.py > blind_review/audit_logs/run_all_audits.log 2>&1
for a in autoresearch_config_audit paper_plan_audit reviewer_caveat_audit scientific_audit; do
  python "${a}.py" > "blind_review/audit_logs/${a}.log" 2>&1 || true
done
```

Both anonymisation scripts are deterministic and idempotent; re-running them
produces the same `paper/main_anon.tex` and a tarball with identical content
(the SHA-256 is stable up to `tarfile` metadata).

---

## 6. Summary

- Anonymised paper produced: `paper/main_anon.tex` — 0 identifier leaks.
- Anonymised code tarball produced: `blind_review/tinker-rl-lab-anon.tar.gz`
  (22,514,454 bytes; SHA-256
  `ef03e4ab3855e33d810a27c73296e90271efd1a4e39283b553d09b53744fb011`;
  539 files; 70 files rewritten; 0 identifier leaks).
- 10 of 13 audits in the main suite pass. The 3 failing audits and the 3
  supplementary audits with issues are all pre-existing and out of scope
  for Task 11 (which forbids modifying the non-anon tree).
- `scientific_audit.py` cannot run in this sandbox (no `pdflatex`); it should
  be re-run on the submission runner.
- The non-anon tree (`paper/main.tex`, `reports/`, repo scripts) was not
  modified.
