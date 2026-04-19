# Blind-Review Submission Manifest — NeurIPS 2026

**Paper:** Tinker RL Lab — An End-to-End Reinforcement-Learning Post-Training
Benchmark for Large Language Models
**Track:** Datasets & Benchmarks (blind review)
**Bundle cut at commit:** (see `INTEGRATION_LOG.md` for SHA after
`task-13-integrator` merges into `main`)
**Date:** 2026-04-19

This manifest lists every file in the blind-review bundle together with its
SHA-256. Any reviewer or program chair can verify bit-for-bit integrity of the
bundle by re-running `sha256sum -c` against this file.

## 1. Included files

| File | Size | SHA-256 |
|---|---|---|
| `main_anon.pdf` | 6,731,307 B | `c5ab440c9d3795b9c994135d75dcdb1b663b2a831856d9dfbaea864e3b254f06` |
| `main_anon.tex` | 104 KB | `da5d87ef2b011f3dd53640162627a847dd83671d27352511b4783791fe4d1143` |
| `tinker-rl-lab-anon.tar.gz` | 27 MB | `5082ce24906b60f3c878b986e1fc36639524d30606bad4a998cc606768b1c976` |
| `anonymize_paper.py` | 20 KB | `1863415492ce266ccd5936c92920eab83e7e94275621b7d548b432f7a5dd8e06` |
| `anonymize_code.py` | 21 KB | `7f32615cb819f49d806dbcd36a81a515e2918150f69ff740d3514011a8b6cb21` |
| `paper_changes.log` | 4.7 KB | `a84f1c3b4cdaa7c1321765c5342223fcac2258c83ef83462e12d3d6bb0aac8d5` |
| `code_changes.log` | 19 KB | `082c64d02d7d83e2742edae92a8996ee4bd9fa3743def1922d1d7c2579f5f74c` |
| `AUDIT.md` | 13 KB | `104afac2ace2ae5b5f750addf2567d3964f3c9a321db5eaf0923a9fa77afabc0` |

## 2. Anonymisation guarantees

1. `main_anon.pdf` was rebuilt from current `main.tex` through
   `blind_review/anonymize_paper.py`, then compiled with
   `latexmk -pdf -interaction=nonstopmode main_anon.tex`. Build is clean
   (0 errors, 0 warnings, 0 overfull/underfull >1pt, 0 undefined references,
   51 pages).
2. `pdftotext main_anon.pdf -` contains zero occurrences of the following
   tokens: `arvindcr4`, `sandhya`, `rafi`, `madhu kumara`, `dhruva`, `arumugam`,
   `anwesh`, `darapaneni`, `narayana`, `PES University`, `Great Learning`,
   `Northwestern University`, `pes-llm-research`, `Balasandhya`, `Madhu2133`,
   `MohammadRafiML`, `dhruvanmurthy`, `madhukumara1993`.
3. `tinker-rl-lab-anon.tar.gz` passes the same regex sweep
   (`blind_review/anonymize_code.py :: _post_scan`) with zero residual
   identifiers.
4. Substitutions are idempotent: re-running either script on the produced
   output is a no-op.

## 3. How to rebuild

```bash
# From repository root
python3 blind_review/anonymize_paper.py
python3 blind_review/anonymize_code.py
cd paper && latexmk -pdf -interaction=nonstopmode main_anon.tex
```

## 4. Verification

```bash
sha256sum -c <<'EOF'
c5ab440c9d3795b9c994135d75dcdb1b663b2a831856d9dfbaea864e3b254f06  main_anon.pdf
da5d87ef2b011f3dd53640162627a847dd83671d27352511b4783791fe4d1143  main_anon.tex
5082ce24906b60f3c878b986e1fc36639524d30606bad4a998cc606768b1c976  tinker-rl-lab-anon.tar.gz
1863415492ce266ccd5936c92920eab83e7e94275621b7d548b432f7a5dd8e06  anonymize_paper.py
7f32615cb819f49d806dbcd36a81a515e2918150f69ff740d3514011a8b6cb21  anonymize_code.py
a84f1c3b4cdaa7c1321765c5342223fcac2258c83ef83462e12d3d6bb0aac8d5  paper_changes.log
082c64d02d7d83e2742edae92a8996ee4bd9fa3743def1922d1d7c2579f5f74c  code_changes.log
104afac2ace2ae5b5f750addf2567d3964f3c9a321db5eaf0923a9fa77afabc0  AUDIT.md
EOF
```
