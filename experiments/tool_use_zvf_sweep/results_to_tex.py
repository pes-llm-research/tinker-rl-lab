#!/usr/bin/env python3
"""
Splice ZVF / pass@1 numbers from a results CSV into the TeX appendix table.

CSV columns (header required):
  model,reward,seed,steps,zvf_mean_last10,pass_at_1
  Qwen3-4B,v1,42,50,0.87,0.31
  Qwen3-4B,v2,42,50,0.52,0.28

Rows already populated (non em-dash) remain intact; rows matching a CSV row
are updated; rows without a matching CSV row stay as em-dashes.

Usage:
  .venv/bin/python experiments/tool_use_zvf_sweep/results_to_tex.py \
      experiments/tool_use_zvf_sweep/results.csv \
      paper/sections/zvf_counterfactual_appendix.tex
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^(?P<pre>\s*(?P<model>Qwen3-4B|Qwen3-8B)\s*&\s*"
    r"(?P<reward>v1|v2)\s*&\s*(?P<seed>\d+)\s*&\s*(?P<steps>\d+)\s*&\s*)"
    r"(?P<zvf>.+?)\s*&\s*(?P<pass>.+?)\s*\\\\\s*$"
)


def _fmt(x: str | float, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "---"


def load_csv(path: Path) -> dict[tuple[str, str, int], dict]:
    rows = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (r["model"].strip(), r["reward"].strip().lower(), int(r["seed"]))
            rows[key] = {
                "zvf": _fmt(r.get("zvf_mean_last10", "")),
                "pass": _fmt(r.get("pass_at_1", ""), digits=2),
            }
    return rows


def splice(csv_path: Path, tex_path: Path) -> None:
    results = load_csv(csv_path) if csv_path.exists() else {}
    tex = tex_path.read_text().splitlines(keepends=True)
    out = []
    filled = 0
    still_missing = []
    for line in tex:
        m = ROW_RE.match(line)
        if not m:
            out.append(line)
            continue
        key = (m.group("model"), m.group("reward"), int(m.group("seed")))
        if key in results:
            r = results[key]
            out.append(f"{m.group('pre')}{r['zvf']} & {r['pass']} \\\\\n")
            filled += 1
        else:
            out.append(line)
            still_missing.append(key)
    tex_path.write_text("".join(out))
    print(f"[splicer] filled {filled} row(s); "
          f"{len(still_missing)} still em-dashed")
    for k in still_missing:
        print(f"  missing: model={k[0]} reward={k[1]} seed={k[2]}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: results_to_tex.py <results.csv> <appendix.tex>")
    splice(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
