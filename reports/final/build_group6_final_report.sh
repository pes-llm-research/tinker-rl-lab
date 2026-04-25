#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_MD="$SCRIPT_DIR/capstone_final_report.md"
OUT_BASE="$SCRIPT_DIR/group6_final_report"
TMP_MD="$(mktemp "${TMPDIR:-/tmp}/group6_final_report.XXXXXX.md")"

cleanup() {
  rm -f "$TMP_MD"
}
trap cleanup EXIT

cat >"$TMP_MD" <<'EOF'
---
title: "Reinforcement Learning for Agentic LLM Fine-Tuning: GRPO-Based Optimization Across Tool Use, Code Generation, and Math Reasoning"
author: "Group 6 | MTech DSAI | PES University"
date: "April 20, 2026"
lang: "en-US"
documentclass: report
classoption:
  - oneside
fontsize: 11pt
geometry:
  - margin=1in
colorlinks: true
linkcolor: blue
urlcolor: blue
papersize: a4
toc: true
toc-depth: 2
mainfont: TeX Gyre Termes
mathfont: Latin Modern Math
monofont: DejaVu Sans Mono
header-includes:
  - \usepackage{booktabs}
---
EOF

# Drop the leading H1 so the report title comes from metadata, then promote the
# remaining headings so the chapter structure survives the PDF/TeX export.
tail -n +3 "$SOURCE_MD" >>"$TMP_MD"

# Normalize a few UI-style status symbols into publication-safe prose while
# keeping the underlying meaning intact.
perl -0pi -e 's/✅/Completed/g; s/✓/Supported/g; s/⚠️/Needs follow-up/g; s/⚠/Needs follow-up/g; s/✗/Not supported/g;' "$TMP_MD"

python - "$TMP_MD" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()

symbol_map = {
    "→": "->",
    "≈": "~",
    "≥": ">=",
    "≤": "<=",
    "≠": "!=",
    "∈": " in ",
    "∀": "for all ",
    "∇": "nabla ",
    "∝": " proportional to ",
    "∧": " and ",
}

sup_map = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "⁺": "+",
    "⁽": "(",
    "⁾": ")",
})

sub_map = str.maketrans({
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₋": "-",
    "₊": "+",
    "₍": "(",
    "₎": ")",
})

for old, new in symbol_map.items():
    text = text.replace(old, new)

text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺⁽⁾]+", lambda m: "^" + m.group(0).translate(sup_map), text)
text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉₋₊₍₎]+", lambda m: "_" + m.group(0).translate(sub_map), text)
text = re.sub(r" {2,}", " ", text)

path.write_text(text)
PY

COMMON_ARGS=(
  --standalone
  --from=markdown+raw_tex+tex_math_dollars+tex_math_single_backslash
  --shift-heading-level-by=-1
  --top-level-division=chapter
)

pandoc "$TMP_MD" \
  "${COMMON_ARGS[@]}" \
  --to=latex \
  --output="$OUT_BASE.tex"

pandoc "$TMP_MD" \
  "${COMMON_ARGS[@]}" \
  --pdf-engine=xelatex \
  --output="$OUT_BASE.pdf"

pandoc "$TMP_MD" \
  "${COMMON_ARGS[@]}" \
  --output="$OUT_BASE.docx"

echo "Built:"
echo "  $OUT_BASE.tex"
echo "  $OUT_BASE.pdf"
echo "  $OUT_BASE.docx"
