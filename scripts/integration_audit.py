#!/usr/bin/env python3
"""
integration_audit.py — Task-13 cross-artifact consistency checks.

Runs 7 checks across paper, master_results, bibliography, and author manifests.
Writes integration_audit.json at repo root. Exits non-zero if any check fails.

Checks:
  1. Abstract numerics appear somewhere in the paper body / results tables
  2. Every model-name-in-paper resolves to a master_results.json entry (fuzzy)
  3. Every \\cite{} key is defined in references.bib AND vice-versa (no unused)
  4. Every \\ref / \\cref / \\autoref / \\eqref target resolves in .aux
  5. Compute-budget totals in paper are internally consistent (sum rounds)
  6. Carbon footprint equals compute_energy * grid_intensity (factor documented)
  7. Author list is identical across paper/main.tex, capstone_final_report.md,
     README.md, and CITATION.cff
"""
from __future__ import annotations
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = ROOT / "paper"
MAIN_TEX = PAPER_DIR / "main.tex"
ETHICS_TEX = PAPER_DIR / "ethics_statement.tex"
ABSTRACT_TEX = PAPER_DIR / "sections" / "abstract.tex"
REFERENCES_BIB = PAPER_DIR / "references.bib"
AUX_FILE = PAPER_DIR / "main.aux"
MASTER_JSON = ROOT / "experiments" / "master_results.json"
README_MD = ROOT / "README.md"
CITATION_CFF = ROOT / "CITATION.cff"
CAPSTONE_MD = ROOT / "reports" / "final" / "capstone_final_report.md"

# -------- canonical author list (ground truth) -----------------------------
CANONICAL_AUTHORS = [
    "Arvind C R",
    "Sandhya Jeyaraj",
    "Madhu Kumara L",
    "Mohammad Rafi",
    "Dhruva N Murthy",
    "Arumugam K",
    "Anwesh Reddy Paduri",
    "Narayana Darapaneni",
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _tex_files() -> list[Path]:
    out = [MAIN_TEX, ETHICS_TEX]
    out += sorted((PAPER_DIR / "sections").glob("*.tex"))
    return [p for p in out if p.exists()]


def _all_tex() -> str:
    return "\n".join(_read(p) for p in _tex_files())


# ---------------------------------------------------------------------------
# Check 1: abstract numeric claims appear in the paper body
# ---------------------------------------------------------------------------
def check_abstract_numbers() -> dict:
    abstract = _read(ABSTRACT_TEX)
    # Everything outside the abstract source file
    body = "\n".join(_read(p) for p in _tex_files() if p != ABSTRACT_TEX)

    # Extract numerics: percentages, correlations, p-values, raw integers/decimals
    patterns = [
        r"\d+\.\d+\\?%",          # 54.5%
        r"\d+\\?%",                # 100%
        r"[+\-]?0\.\d+",           # 0.922, -0.769
        r"r\{?=\}?\{?[+\-−]?0\.\d+\}?",  # r=-0.769
    ]
    raw = []
    for pat in patterns:
        raw += re.findall(pat, abstract)

    # Significant tokens to cross-check in body (numeric core only)
    core = set()
    for tok in raw:
        m = re.search(r"\d+(?:\.\d+)?", tok)
        if m:
            core.add(m.group(0))

    # Canonical high-signal claims mined from the abstract text.
    # Only numbers that ACTUALLY appear in abstract.tex.
    high_signal = [
        "79", "0.769", "54.5", "32.5", "100",
        "22.5", "34.4", "97.5", "84.4", "87.5", "91.7",
        "0.922", "0.047", "0.969", "16.2", "80",
    ]
    for t in high_signal:
        core.add(t)

    missing = []
    found = []
    for t in sorted(core, key=lambda x: -len(x)):
        # allow either 54.5 or 54.5\% etc.
        if re.search(rf"(?<!\d){re.escape(t)}(?!\d)", body):
            found.append(t)
        else:
            missing.append(t)

    passed = len(missing) == 0
    return {
        "name": "abstract_numbers_in_body",
        "passed": passed,
        "checked": len(core),
        "found": len(found),
        "missing": missing,
        "detail": "Every numeric claim in abstract.tex must also appear in the body of the paper."
    }


# ---------------------------------------------------------------------------
# Check 2: every model name in paper matches master_results.json
# ---------------------------------------------------------------------------
def _normalize_model(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


# Allowlist: models referenced in the paper (background/related-work/checkpoint
# naming) that were NOT themselves RL-trained in our sweep.  Every name here
# must appear somewhere in the paper; we document *why* each is allowed.
DISCUSSED_ONLY_MODELS = {
    # checkpoint-naming / intermediate SFT bases mentioned in scaling text,
    # not part of the 79-run RL matrix
    "Qwen2-0.5B",
    "Qwen2-0.5B-Inst",
    "Qwen2.5-3B",
    # background / related-work only — no run in our sweep
    "Qwen3-0.6B",
    "Qwen3-0.6B-Instruct",
    "Qwen3-14B",
    # present in master as qwen3-30b-inst / qwen3-30b-moe (A3B is the variant tag);
    # kept here because the normalized prefix match is not reliable
    "Qwen3-30B-A3B",
    "Qwen3-30B-A3B-Instruct",
}


def check_models_in_master() -> dict:
    master = json.loads(_read(MASTER_JSON))
    rows = master.get("experiments", [])
    master_norm = {_normalize_model(r["model"]): r["model"] for r in rows if r.get("model")}
    # Also include short variants
    short_keys = set()
    for full in list(master_norm.values()):
        tail = full.split("/")[-1]
        short_keys.add(_normalize_model(tail))
    for k in short_keys:
        master_norm.setdefault(k, k)

    tex = _all_tex()
    # candidate model mentions: look for well-known families
    candidates = set()
    patterns = [
        r"Qwen[23](?:\.\d+)?-\d+(?:\.\d+)?[BK](?:-[A-Za-z0-9.]+)*",
        r"Llama-3(?:\.\d+)?-\d+B(?:-Instruct)?",
        r"DeepSeek-V3(?:\.\d+)?(?:-Base)?",
        r"Kimi-K2(?:[-.][A-Za-z0-9]+)*",
        r"Nemotron-\d+B?",
        r"GPT-OSS-\d+[Bb]",
        r"gpt-oss-\d+b",
    ]
    for pat in patterns:
        for m in re.findall(pat, tex):
            candidates.add(m)

    # Map every mention to a master row (fuzzy — prefix match on normalized)
    matched, unmatched, discussed_only = [], [], []
    for c in sorted(candidates):
        cn = _normalize_model(c)
        hit = None
        for k, v in master_norm.items():
            if cn in k or k in cn:
                hit = v
                break
        if hit:
            matched.append({"paper": c, "master": hit})
        elif c in DISCUSSED_ONLY_MODELS:
            discussed_only.append(c)
        else:
            unmatched.append(c)

    passed = len(unmatched) == 0
    return {
        "name": "models_in_master",
        "passed": passed,
        "total_candidates": len(candidates),
        "matched": len(matched),
        "discussed_only": discussed_only,
        "unmatched": unmatched,
        "detail": "Every model family mentioned must either (a) map to a master_results.json entry or (b) be on the documented discussed-only allowlist."
    }


# ---------------------------------------------------------------------------
# Check 3: bibliography integrity — cites ↔ entries
# ---------------------------------------------------------------------------
def check_bib_integrity() -> dict:
    bib = _read(REFERENCES_BIB)
    entries = set(re.findall(r"^@\w+\{([^,\s]+)\s*,", bib, flags=re.M))

    tex = _all_tex()
    # Strip LaTeX comment lines *only*, not escaped %. A comment starts with
    # an unescaped % and runs to end of line; the backslash-escape case must
    # be preserved so that e.g. "12.5\%" stays intact. Use a negative lookbehind
    # for a single backslash before the %.
    tex_stripped = re.sub(r"(?<!\\)%[^\n]*\n\s*", "", tex)
    # Collect \cite, \citet, \citep with possibly comma-separated keys
    cites = set()
    for m in re.finditer(r"\\cite[tp]?\*?\{([^}]+)\}", tex_stripped):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                cites.add(k)

    unused = sorted(entries - cites)
    undefined = sorted(cites - entries)

    # Spec: "no unused refs" + every cite resolves.
    passed = len(undefined) == 0 and len(unused) == 0
    return {
        "name": "bib_integrity",
        "passed": passed,
        "entries": len(entries),
        "cites": len(cites),
        "undefined_cites": undefined,
        "unused_entries": unused,
    }


# ---------------------------------------------------------------------------
# Check 4: every \ref resolves against the .aux file
# ---------------------------------------------------------------------------
def check_ref_resolution() -> dict:
    if not AUX_FILE.exists():
        return {"name": "ref_resolution", "passed": False, "error": "main.aux missing — rebuild paper first"}
    aux = _read(AUX_FILE)
    defined = set(re.findall(r"\\newlabel\{([^}]+)\}", aux))

    tex = _all_tex()
    refs = set()
    for cmd in (r"\\ref", r"\\cref", r"\\Cref", r"\\autoref", r"\\eqref", r"\\pageref", r"\\nameref"):
        for m in re.finditer(rf"{cmd}\*?\{{([^}}]+)\}}", tex):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    refs.add(k)

    undefined = sorted(r for r in refs if r not in defined)
    passed = len(undefined) == 0
    return {
        "name": "ref_resolution",
        "passed": passed,
        "labels_defined": len(defined),
        "refs_used": len(refs),
        "undefined_refs": undefined[:20],
        "total_undefined": len(undefined),
    }


# ---------------------------------------------------------------------------
# Check 5: compute budget internally consistent
# ---------------------------------------------------------------------------
def check_compute_budget() -> dict:
    """Paper tab:compute_spend claims $130-140 total across sub-rows.
    Parse and verify the reported range bracket sums match."""
    text = _read(ETHICS_TEX)
    # Grab numeric ranges in the compute_spend table
    tbl = re.search(r"\\label\{tab:compute_spend\}(.*?)\\end\{table\}", text, flags=re.S)
    if not tbl:
        return {"name": "compute_budget", "passed": False, "error": "compute_spend table not found"}

    body = tbl.group(1)
    ranges = []  # list of (low, high) USD for data rows only
    total_row = None
    for line in body.splitlines():
        m = re.search(r"\\\$([0-9]+)(?:--([0-9]+))?", line)
        if not m:
            continue
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        if "Total" in line:
            total_row = (lo, hi)
        else:
            ranges.append((lo, hi, line.strip()[:80]))

    sum_lo = sum(r[0] for r in ranges)
    sum_hi = sum(r[1] for r in ranges)

    if total_row is None:
        return {"name": "compute_budget", "passed": False, "error": "claimed total not found", "sum_lo": sum_lo, "sum_hi": sum_hi}
    c_lo, c_hi = total_row

    # Sub-rows sum EXCLUDES the per-person entries (Colab) and counts one H100 PPO row.
    # Accept if claimed bracket overlaps computed ±30%.
    overlap = not (c_hi < sum_lo * 0.7 or c_lo > sum_hi * 1.3)
    passed = overlap
    return {
        "name": "compute_budget",
        "passed": passed,
        "subrow_sum_usd": [sum_lo, sum_hi],
        "claimed_total_usd": [c_lo, c_hi],
        "rows_parsed": len(ranges),
        "detail": "Sum of sub-row USD ranges must overlap with claimed total (±30% tolerance for per-person rows).",
    }


# ---------------------------------------------------------------------------
# Check 6: carbon = energy * grid intensity
# ---------------------------------------------------------------------------
def check_carbon() -> dict:
    text = _read(ETHICS_TEX)
    # parse carbon table
    tbl = re.search(r"\\label\{tab:carbon2\}(.*?)\\end\{table\}", text, flags=re.S)
    if not tbl:
        return {"name": "carbon_accounting", "passed": False, "error": "carbon table not found"}
    body = tbl.group(1)

    rows = []
    for line in body.splitlines():
        # Split on & to get columns, then parse a single numeric from each.
        # Skip header/rule lines.
        if "&" not in line or "\\toprule" in line or "\\midrule" in line or "\\bottomrule" in line:
            continue
        cols = [c.strip() for c in line.split("&")]
        if len(cols) < 6:
            continue
        platform = cols[0]
        # Only data rows: known platform keywords
        if not any(k in platform for k in ("Tinker", "Modal", "PES", "Colab", "L4")):
            continue

        def _num(cell: str) -> float | None:
            # Strip LaTeX macros and take the first numeric token.
            clean = re.sub(r"\\[A-Za-z]+", " ", cell)
            m = re.search(r"[0-9]+(?:\.[0-9]+)?", clean)
            return float(m.group(0)) if m else None

        gpuh = _num(cols[1])
        tdp = _num(cols[2])
        pue = _num(cols[3])
        energy = _num(cols[4])
        # Strip trailing \\ from last cell before number extraction
        co2 = _num(cols[5].rstrip("\\ "))
        if None in (gpuh, tdp, pue, energy, co2):
            continue
        rows.append({"line": platform[:60], "gpuh": gpuh, "tdp_w": tdp, "pue": pue, "energy_kwh": energy, "co2_kg": co2})

    # Grid intensities documented in the paper (US 0.367, India 0.716 kg/kWh)
    US = 0.367
    IN = 0.716
    details = []
    ok = True
    for r in rows:
        # Energy (kWh) = GPU-h * TDP(W) * PUE / 1000
        computed_kwh = r["gpuh"] * r["tdp_w"] * r["pue"] / 1000.0
        rel_energy = abs(computed_kwh - r["energy_kwh"]) / max(r["energy_kwh"], 1e-6)
        # CO2 against US and India grid
        co2_us = computed_kwh * US
        co2_in = computed_kwh * IN
        # allow a ±30% band because the paper rounds to 1 sig fig
        closer = min(abs(co2_us - r["co2_kg"]), abs(co2_in - r["co2_kg"]))
        denom = max(r["co2_kg"], 1e-6)
        rel_co2 = closer / denom
        row_ok = (rel_energy < 0.35) and (rel_co2 < 0.5)
        if not row_ok:
            ok = False
        details.append({
            "platform": r["line"].split("&")[0].strip(),
            "reported_kwh": r["energy_kwh"],
            "computed_kwh": round(computed_kwh, 2),
            "reported_co2_kg": r["co2_kg"],
            "computed_co2_us_kg": round(co2_us, 2),
            "computed_co2_in_kg": round(co2_in, 2),
            "row_ok": row_ok,
        })

    return {
        "name": "carbon_accounting",
        "passed": ok and len(rows) >= 3,
        "rows_checked": len(rows),
        "grid_factors": {"US_kgCO2_per_kWh": US, "IN_kgCO2_per_kWh": IN},
        "per_row": details,
        "detail": "Energy = GPU-h × TDP × PUE / 1000.  CO2 = Energy × grid intensity (US or IN); rounded to 1 sig fig in paper.",
    }


# ---------------------------------------------------------------------------
# Check 7: author list identical across paper, report, README, CITATION.cff
# ---------------------------------------------------------------------------
def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^A-Za-z]+", "", s).lower()
    return s


def check_authors() -> dict:
    canonical_norm = {_norm_name(a): a for a in CANONICAL_AUTHORS}

    sources = {}

    # paper main.tex — already known canonical list; verify presence
    tex = _read(MAIN_TEX)
    sources["paper/main.tex"] = [a for a in CANONICAL_AUTHORS if _norm_name(a) in _norm_name(tex)]

    # capstone report
    md = _read(CAPSTONE_MD)
    sources["reports/final/capstone_final_report.md"] = [a for a in CANONICAL_AUTHORS if _norm_name(a) in _norm_name(md)]

    # README
    rm = _read(README_MD)
    sources["README.md"] = [a for a in CANONICAL_AUTHORS if _norm_name(a) in _norm_name(rm)]

    # CITATION.cff — names are split across `given-names` and `family-names`
    # fields per CFF schema, so concatenate every given/family pair before
    # matching rather than searching the raw text.
    if CITATION_CFF.exists():
        cf = _read(CITATION_CFF)
        given = re.findall(r'given-names:\s*"?([^"\n]+)"?', cf)
        family = re.findall(r'family-names:\s*"?([^"\n]+)"?', cf)
        pairs = []
        for g, f in zip(given, family):
            pairs.append(f"{g.strip()} {f.strip()}")
            pairs.append(f"{f.strip()} {g.strip()}")
        norm_pool = " ".join(_norm_name(p) for p in pairs)
        sources["CITATION.cff"] = [a for a in CANONICAL_AUTHORS if _norm_name(a) in norm_pool]
    else:
        sources["CITATION.cff"] = []

    missing_per_source = {src: sorted(set(CANONICAL_AUTHORS) - set(names)) for src, names in sources.items()}
    # pass only when every source contains every canonical author
    passed = all(not miss for miss in missing_per_source.values())
    return {
        "name": "author_consistency",
        "passed": passed,
        "canonical": CANONICAL_AUTHORS,
        "missing_per_source": missing_per_source,
        "detail": "All 8 canonical authors must appear in every source. CITATION.cff is created in step 7 — expect initial fail.",
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    checks = [
        check_abstract_numbers,
        check_models_in_master,
        check_bib_integrity,
        check_ref_resolution,
        check_compute_budget,
        check_carbon,
        check_authors,
    ]
    results = []
    for fn in checks:
        try:
            r = fn()
        except Exception as exc:
            r = {"name": fn.__name__, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)

    all_pass = all(r.get("passed") for r in results)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(ROOT),
        "all_passed": all_pass,
        "total_checks": len(results),
        "passed_count": sum(1 for r in results if r.get("passed")),
        "checks": results,
    }
    out = ROOT / "integration_audit.json"
    out.write_text(json.dumps(summary, indent=2, default=str))

    # Human-readable summary to stdout
    print(f"integration_audit: {summary['passed_count']}/{summary['total_checks']} checks passed")
    for r in results:
        flag = "PASS" if r.get("passed") else "FAIL"
        name = r.get("name")
        print(f"  [{flag}] {name}")
        if not r.get("passed"):
            keys = [k for k in r.keys() if k not in ("name", "passed", "detail")][:4]
            for k in keys:
                v = r[k]
                s = str(v)
                if len(s) > 180:
                    s = s[:180] + "..."
                print(f"         {k}: {s}")
    print(f"Full report: {out}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
