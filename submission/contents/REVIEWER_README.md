# Tinker RL Lab — NeurIPS 2026 (Datasets & Benchmarks, blind review)

Welcome, reviewer. This bundle contains everything you need to assess
the submission.

## Files

| File | What it is |
|---|---|
| `paper_anon.pdf` | **Anonymised camera-ready paper (51 pages).** This is the version to review. |
| `paper.pdf` | Non-anonymous paper, included for ACs only. Do not open unless needed for de-anonymisation checks. |
| `code.tar.gz` | Anonymised code tarball (27 MB, 643 files). Extract and follow `REPRODUCE.md` inside. |
| `ethics_statement.pdf` | Standalone ethics statement (duplicates §Ethics in the paper). |
| `data_statement.md` | Dataset provenance, licensing, PII / offensive-content notes. |
| `MANIFEST.md` | SHA-256 of every file in this bundle. |
| `REVIEWER_README.md` | This file. |

## How to run the reproducibility smoke test

```bash
tar xzf code.tar.gz
cd tinker-rl-lab-anon
# Offline test subset (~25 s CPU):
python3 -m pytest reproducibility/test_offline_subset.py -v
# Headline-claim check (±2 pp tolerance):
python3 reproducibility/check_qwen3_8b_claim.py
```

`scripts/integration_audit.py` reproduces the 7 integration checks we
used to verify paper↔code↔data consistency.

## Anonymity guarantees

- `paper_anon.pdf` passes `pdftotext | grep` for every team member
  name, handle, institution, and private repo slug with zero hits.
- `code.tar.gz` passes the identifier post-scan in
  `blind_review/anonymize_code.py` with zero residuals.
- Substitutions are idempotent: you can re-run the anonymisation
  scripts on the produced output and get a no-op.

## Contact

Please use the OpenReview submission thread for questions. We will
not be able to respond on GitHub because the public mirror is
de-anonymising.
