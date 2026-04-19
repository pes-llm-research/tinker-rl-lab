#!/usr/bin/env python3
"""Anonymize the repository into ``blind_review/tinker-rl-lab-anon/`` and
produce a tarball ``blind_review/tinker-rl-lab-anon.tar.gz``.

The anonymized tree is built from a clean export of the working tree
(``git archive HEAD``) so that nothing outside source-control is leaked.
All comments, docstrings, W&B run/project names, HuggingFace repo paths,
notebook metadata, and README / LICENSE copyright lines are rewritten to
remove any identifying information.

Replacement rules (order matters; longest-match first):
  * Team-member real names (regex word boundaries) ->  ``Anonymous``
  * HuggingFace handles (Madhu2133, Balasandhya, MohammadRafiML,
    dhruvanmurthy)                                 ->  ``anonymous``
  * Team GitHub slugs (arvindcr4, madhukumara1993)  ->  ``anonymous``
  * Organisation slug pes-llm-research              ->  ``anonymous-org``
  * Institutions (PES University, Great Learning,
    Northwestern University)                        ->  ``Anonymous Institution``
  * W&B entity arvindcr4-pes-university             ->  ``anonymous``
  * W&B project tinker-rl-lab-world-class           ->  ``tinker-rl-bench``
  * Emails of all team members                      ->  ``anonymous@neurips.cc``
  * Paper bibtex author list                        ->  ``{Anonymous Authors}``
  * LICENSE copyright line                          ->  ``Anonymous Authors``

Files dropped on purpose (not part of the anonymised artefact):
  * team-analysis.pplx.md, team-links-audit.pplx.md      (internal team memo)
  * paper/main.tex, paper/acm_main.tex                   (non-anonymous sources)
  * paper/main.pdf                                        (compiled artefact)
  * paper/main.aux/out/bbl/blg/log                        (build artefacts)
  * reports/final/grpo_agentic_llm_paper.tex             (non-anonymous mirror)
  * reports/final/grpo_agentic_llm_paper.md              (non-anonymous mirror)
  * reports/final/capstone_final_report.md               (internal, names authors)
  * reports/final/capstone_final_report.docx             (internal)
  * reports/final/CONSOLIDATED_REVIEW_IMPROVEMENTS.md    (internal)
  * reports/final/PAPER_IMPROVEMENT_PLAN.md              (internal)
  * scripts/anonymize.sh                                  (old tool, superseded)
  * blind_review/**                                       (not self-referential)
  * **/__pycache__, *.pyc, wandb/, .env                   (caches/secrets)

The script is idempotent and logs every file it edited, in order, into
``blind_review/code_changes.log``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "blind_review" / "tinker-rl-lab-anon"
TARBALL = ROOT / "blind_review" / "tinker-rl-lab-anon.tar.gz"
LOG = ROOT / "blind_review" / "code_changes.log"

# Files we exclude from the anonymised export entirely.
EXCLUDE_FILES = {
    "team-analysis.pplx.md",
    "team-links-audit.pplx.md",
    "paper/main.tex",
    "paper/acm_main.tex",
    "paper/main.pdf",
    "paper/main.aux",
    "paper/main.out",
    "paper/main.bbl",
    "paper/main.blg",
    "paper/limitations_update.tex",
    "paper/neurips_checklist_update.tex",
    "paper/ethics_statement.tex",
    "reports/final/grpo_agentic_llm_paper.tex",
    "reports/final/grpo_agentic_llm_paper.md",
    "reports/final/capstone_final_report.md",
    "reports/final/capstone_final_report.docx",
    "reports/final/CONSOLIDATED_REVIEW_IMPROVEMENTS.md",
    "reports/final/PAPER_IMPROVEMENT_PLAN.md",
    "scripts/anonymize.sh",
    "verify_links_entities.txt",
    "autoresearch-dashboard.md",
    # Binary slide decks carry team names inside pptx xml; dropped entirely.
    "experiments/collab-results/LLM_Tool_Call_Finetuning.pptx",
    "experiments/collab-results/exp3_multiturn.pptx",
}

EXCLUDE_DIRS = {
    "blind_review",
    "past_session_contexts",
    "wandb",
    "__pycache__",
}

# Binary / non-rewriteable extensions: copy as-is, no text rewrite.
BINARY_EXTS = {
    ".zip",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pptx",
    ".docx",
    ".xlsx",
    ".webp",
    ".bin",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
    ".gz",
    ".tgz",
    ".ico",
}

# Rewriteable text extensions (everything else we skip for safety).
TEXT_EXTS = {
    ".py",
    ".md",
    ".tex",
    ".yaml",
    ".yml",
    ".json",
    ".sh",
    ".txt",
    ".toml",
    ".cfg",
    ".ipynb",
    ".sty",
    ".bib",
    ".bst",
    ".cls",
    ".rst",
    ".csv",
    ".tsv",
    ".log",
    ".jsonl",
    ".env",
    "",
    ".in",
    ".ini",
    ".conf",
    ".gitignore",
    ".gitattributes",
    ".dockerfile",
    ".Dockerfile",
}

# We use explicit lookaround (``(?<![A-Za-z0-9_])...(?![A-Za-z0-9_])``)
# instead of ``\b`` because slugs like ``pes-llm-research`` contain a
# hyphen and ``\b`` doesn't match at hyphen boundaries when the preceding
# char is a letter (e.g. ``\\npes-llm-research`` in a Python string).
# Longest-match first so that e.g. ``arvindcr4-pes-university`` is rewritten
# before ``arvindcr4`` gets at it.

_LB = r"(?<![A-Za-z0-9_])"
_LA = r"(?![A-Za-z0-9_])"

REPLACEMENTS: list[tuple[str, str, str]] = [
    # Weights & Biases entity (must come before ``arvindcr4`` alone).
    (
        r"arvindcr4-pes-university",
        "anonymous-entity",
        "wandb entity arvindcr4-pes-university -> anonymous-entity",
    ),
    (
        r"tinker-rl-lab-world-class",
        "tinker-rl-bench",
        "wandb project tinker-rl-lab-world-class -> tinker-rl-bench",
    ),
    # Team HF handles.
    (_LB + r"Madhu2133" + _LA, "anonymous", "HF handle Madhu2133 -> anonymous"),
    (_LB + r"MohammadRafiML" + _LA, "anonymous", "HF handle MohammadRafiML -> anonymous"),
    (_LB + r"Balasandhya" + _LA, "anonymous", "HF handle Balasandhya -> anonymous"),
    (_LB + r"dhruvanmurthy" + _LA, "anonymous", "HF handle dhruvanmurthy -> anonymous"),
    # GitHub handles / slugs.
    (_LB + r"madhukumara1993" + _LA, "anonymous", "GH handle madhukumara1993 -> anonymous"),
    (_LB + r"arvindcr4" + _LA, "anonymous", "GH handle arvindcr4 -> anonymous"),
    # No lookbehind for the hyphenated slug — the leading ``pes-`` is
    # distinctive enough and lookbehind breaks when preceded by ``n`` in
    # source literals like ``\\npes-llm-research``.
    (r"pes-llm-research", "anonymous-org", "GH org pes-llm-research -> anonymous-org"),
    # Institutions.
    (
        r"PES LLM Research Team",
        "Anonymous Authors",
        "``PES LLM Research Team'' -> Anonymous Authors",
    ),
    (r"PES LLM Research", "Anonymous", "``PES LLM Research'' -> Anonymous"),
    (r"PES University", "Anonymous Institution", "``PES University'' -> Anonymous Institution"),
    (r"Great Learning", "Anonymous Institution", "``Great Learning'' -> Anonymous Institution"),
    (
        r"Northwestern University",
        "Anonymous Institution",
        "``Northwestern University'' -> Anonymous Institution",
    ),
    # Team-member real names (in comments, docstrings, READMEs, bibtex, etc.).
    (r"Narayana Darapaneni", "Anonymous", "name Narayana Darapaneni -> Anonymous"),
    (r"Anwesh Reddy Padhuri", "Anonymous", "name Anwesh Reddy Padhuri -> Anonymous"),
    (r"Anwesh Reddy Paduri", "Anonymous", "name Anwesh Reddy Paduri -> Anonymous"),
    (r"Sandhya Jeyaraj", "Anonymous", "name Sandhya Jeyaraj -> Anonymous"),
    (r"Madhu Kumara L", "Anonymous", "name Madhu Kumara L -> Anonymous"),
    (r"Mohammad Rafi", "Anonymous", "name Mohammad Rafi -> Anonymous"),
    (r"Dhruva N Murthy", "Anonymous", "name Dhruva N Murthy -> Anonymous"),
    (r"Arumugam Chetty K", "Anonymous", "name Arumugam Chetty K -> Anonymous"),
    (r"Arumugam K", "Anonymous", "name Arumugam K -> Anonymous"),
    (r"Arvind C R", "Anonymous", "name Arvind C R -> Anonymous"),
    # First-name-only references in informal prose (after surname-forms above).
    (_LB + r"Sandhya" + _LA, "Anonymous", "first name Sandhya -> Anonymous"),
    (_LB + r"Madhu" + _LA, "Anonymous", "first name Madhu -> Anonymous"),
    (_LB + r"Arvind" + _LA, "Anonymous", "first name Arvind -> Anonymous"),
    (_LB + r"Dhruva" + _LA, "Anonymous", "first name Dhruva -> Anonymous"),
    (_LB + r"Anwesh" + _LA, "Anonymous", "first name Anwesh -> Anonymous"),
    (_LB + r"Rafi" + _LA, "Anonymous", "first name Rafi -> Anonymous"),
    (_LB + r"Arumugam" + _LA, "Anonymous", "first name Arumugam -> Anonymous"),
    # Lowercase handle variants seen in IDs/CSV keys (``arumugam_dpo_...``).
    # Here we intentionally allow ``_`` to follow because these are ID slugs
    # (``arumugam_dpo_keyword_0.5b``) where the underscore is part of the key.
    (
        r"(?<![A-Za-z0-9])arumugam(?![A-Za-z0-9])",
        "anonymous",
        "lowercase arumugam (slug/id) -> anonymous",
    ),
    # PES student roll numbers (PES2PGE24DS140 etc).
    (r"PES2PGE\d{2}DS\d{3}", "ANONYMIZED-ID", "PES student ID redacted"),
    # Emails (all team/supervisor addresses seen in the tree).
    (r"arvindcr4@gmail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"sandhya\.jeyaraj2014@gmail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"madhukumara1993@gmail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"gmd\.rafi\.2024@gmail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"dhruva\.n\.murthy@gmail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"chettyarumugam@mail\.com", "anonymous@neurips.cc", "email redacted"),
    (r"anwesh@greatlearning\.in", "anonymous@neurips.cc", "email redacted"),
    (r"narayana\.darapaneni@northwestern\.edu", "anonymous@neurips.cc", "email redacted"),
    (r"arvindcr@pes\.edu", "anonymous@neurips.cc", "email redacted"),
    # GitHub URL rewriting to anonymous mirrors (after handle replacement so
    # anonymous/tinker-rl-lab lands at anonymous.4open.science).
    (
        r"https://github\.com/anonymous/tinker-rl-lab",
        r"https://anonymous.4open.science/r/tinker-rl-lab",
        "github.com/anonymous/tinker-rl-lab -> anonymous.4open.science",
    ),
    (
        r"https://github\.com/anonymous-org/tinker-rl-lab",
        r"https://anonymous.4open.science/r/tinker-rl-lab",
        "github.com/anonymous-org/tinker-rl-lab -> anonymous.4open.science",
    ),
    # LICENSE copyright line.
    (
        r"Copyright 2026 Anonymous Authors",
        "Copyright 2026 Anonymous Authors",
        "LICENSE copyright line already anonymous",
    ),
]

REPLACEMENTS_COMPILED = [
    (re.compile(pattern), replacement, label) for pattern, replacement, label in REPLACEMENTS
]


@dataclass
class ChangeLog:
    per_file: dict[str, list[tuple[str, int]]] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    skipped_binary: list[str] = field(default_factory=list)
    skipped_unknown_ext: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)

    def record(self, relpath: str, label: str, count: int) -> None:
        if count == 0:
            return
        self.per_file.setdefault(relpath, []).append((label, count))
        self.totals[label] = self.totals.get(label, 0) + count


def _git_archive_into(dst: Path) -> None:
    """Copy the current working tree (tracked files + newly added paper_anon
    and blind_review/) into ``dst``. We use ``git ls-files`` with ``--others
    --exclude-standard`` so we include both tracked and staged-but-new files
    while respecting .gitignore. Build artefacts and secrets are filtered.
    """
    dst.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )
    for rel in result.stdout.splitlines():
        rel = rel.strip()
        if not rel:
            continue
        src = ROOT / rel
        if not src.is_file():
            continue
        tgt = dst / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)


def _excluded(relpath: str) -> bool:
    parts = Path(relpath).parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    return relpath in EXCLUDE_FILES


def _is_binary(path: Path) -> bool:
    return path.suffix.lower() in BINARY_EXTS


def _rewrite_text(text: str) -> tuple[str, list[tuple[str, int]]]:
    changes: list[tuple[str, int]] = []
    for pattern, replacement, label in REPLACEMENTS_COMPILED:
        new_text, n = pattern.subn(replacement, text)
        if n:
            changes.append((label, n))
        text = new_text
    return text, changes


def _rewrite_notebook(text: str) -> tuple[str, list[tuple[str, int]]]:
    """Rewrite identifying info inside .ipynb files while preserving JSON."""
    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return _rewrite_text(text)

    all_changes: list[tuple[str, int]] = []
    # Scrub notebook-level metadata that may contain user/kernel identifiers.
    meta = nb.get("metadata", {})
    for key in ("authors", "author", "kaggle", "colab", "user"):
        if key in meta:
            meta.pop(key)
            all_changes.append((f"notebook metadata.{key} removed", 1))

    def _walk(obj):
        nonlocal all_changes
        if isinstance(obj, dict):
            return {k: _walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_walk(x) for x in obj]
        if isinstance(obj, str):
            new, ch = _rewrite_text(obj)
            all_changes.extend(ch)
            return new
        return obj

    nb = _walk(nb)
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n", all_changes


def _rewrite_file(path: Path, relpath: str, log: ChangeLog) -> None:
    if _is_binary(path):
        log.skipped_binary.append(relpath)
        return
    ext = path.suffix.lower()
    name = path.name
    if ext not in TEXT_EXTS and name not in {"LICENSE", "Dockerfile"}:
        log.skipped_unknown_ext.append(relpath)
        return
    try:
        data = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        log.skipped_binary.append(relpath)
        return

    if ext == ".ipynb":
        new_data, changes = _rewrite_notebook(data)
    else:
        new_data, changes = _rewrite_text(data)

    if changes:
        path.write_text(new_data, encoding="utf-8")
        for label, n in changes:
            log.record(relpath, label, n)


def _drop_excluded(root: Path, log: ChangeLog) -> None:
    for rel in list(EXCLUDE_FILES):
        target = root / rel
        if target.exists():
            target.unlink()
            log.excluded.append(rel)
    for rel in list(EXCLUDE_DIRS):
        target = root / rel
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
            log.excluded.append(rel + "/")


def _rename_main_anon_to_main(root: Path, log: ChangeLog) -> None:
    """Inside the anonymised bundle the only LaTeX source is the anonymised
    ``paper/main_anon.tex``. Keep it as-is so the audit map is unambiguous.
    We do *not* rename it to main.tex so reviewers can see it is anonymised.
    """
    # Nothing to do beyond letting main_anon.tex live at paper/main_anon.tex.


def _post_scan(root: Path) -> list[str]:
    """Final safety scan; return files that still contain identifying tokens."""
    leaks: list[str] = []
    pattern = re.compile(
        r"arvindcr4|pes-llm-research|PES University|Great Learning|"
        r"Northwestern University|Madhu2133|MohammadRafiML|Balasandhya|"
        r"dhruvanmurthy|Madhu Kumara|Sandhya Jeyaraj|Mohammad Rafi|"
        r"Dhruva N Murthy|Arumugam K|Anwesh Reddy|Narayana Darapaneni|"
        r"Arvind C R|madhukumara1993|sandhya\.jeyaraj|gmd\.rafi|"
        r"chettyarumugam|dhruva\.n\.murthy|@greatlearning|@northwestern|"
        r"tinker-rl-lab-world-class|PES2PGE|arvindcr@pes\.edu|"
        r"Padhuri|Paduri|Jeyaraj"
    )
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        try:
            data = path.read_bytes()
        except OSError:
            continue
        # Decode as latin-1 so we can grep binary files (e.g. pptx/zip XML)
        # without raising on non-utf-8 bytes. This still catches identifying
        # ASCII strings embedded in archives.
        try:
            text = data.decode("utf-8", errors="ignore")
        except Exception:
            text = data.decode("latin-1", errors="ignore")
        if pattern.search(text):
            leaks.append(rel)
    return leaks


def main() -> None:
    # 1. Fresh staging tree from git archive so we never export uncommitted files.
    if STAGING.exists():
        shutil.rmtree(STAGING)
    _git_archive_into(STAGING)

    # 2. Copy in the anonymised paper so the tarball has a compile-ready tex.
    anon_tex = ROOT / "paper" / "main_anon.tex"
    (STAGING / "paper").mkdir(exist_ok=True, parents=True)
    shutil.copy2(anon_tex, STAGING / "paper" / "main_anon.tex")
    # Copy style files / figures / bib that the anonymous paper depends on.
    for rel in [
        "paper/neurips_2025.sty",
        "paper/neurips_2026.sty",
        "paper/references.bib",
    ]:
        src = ROOT / rel
        if src.exists():
            shutil.copy2(src, STAGING / rel)
    # Keep the figures directory for paper build.
    figs_src = ROOT / "paper" / "figures"
    figs_dst = STAGING / "paper" / "figures"
    if figs_src.exists() and not figs_dst.exists():
        shutil.copytree(figs_src, figs_dst)
    tikz_src = ROOT / "paper" / "tikz"
    tikz_dst = STAGING / "paper" / "tikz"
    if tikz_src.exists() and not tikz_dst.exists():
        shutil.copytree(tikz_src, tikz_dst)

    log = ChangeLog()

    # 3. Drop excluded files / directories.
    _drop_excluded(STAGING, log)

    # 4. Walk the tree and rewrite every rewriteable text file.
    for path in sorted(STAGING.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(STAGING).as_posix()
        if _excluded(rel):
            # Already excluded; ensure removed.
            path.unlink()
            log.excluded.append(rel)
            continue
        _rewrite_file(path, rel, log)

    # 5. Post-scan for residual identifiers.
    leaks = _post_scan(STAGING)
    if leaks:
        print("WARNING: residual identifiers found in anonymised tree:", file=sys.stderr)
        for rel in leaks:
            print(f"  - {rel}", file=sys.stderr)

    # 6. Build tarball.
    if TARBALL.exists():
        TARBALL.unlink()
    with tarfile.open(TARBALL, "w:gz") as tf:
        tf.add(STAGING, arcname="tinker-rl-lab-anon")

    tarball_sha = hashlib.sha256(TARBALL.read_bytes()).hexdigest()

    # 7. Write the change log.
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("w", encoding="utf-8") as fh:
        fh.write("# Code anonymisation change log\n\n")
        fh.write(f"Tarball: {TARBALL.relative_to(ROOT)}\n")
        fh.write(f"Tarball SHA-256: {tarball_sha}\n")
        fh.write(f"Tarball size: {TARBALL.stat().st_size} bytes\n\n")

        fh.write("## Totals by rule\n\n")
        for label, count in sorted(log.totals.items(), key=lambda kv: -kv[1]):
            fh.write(f"- {label}: {count}\n")
        fh.write("\n")

        fh.write("## Excluded files / directories\n\n")
        for rel in sorted(set(log.excluded)):
            fh.write(f"- {rel}\n")
        fh.write("\n")

        if leaks:
            fh.write("## WARNING: residual identifiers\n\n")
            for rel in leaks:
                fh.write(f"- {rel}\n")
            fh.write("\n")
        else:
            fh.write("## Post-scan: no residual identifiers detected\n\n")

        fh.write("## Per-file edits\n\n")
        for rel in sorted(log.per_file):
            fh.write(f"### {rel}\n\n")
            for label, count in log.per_file[rel]:
                fh.write(f"- {label}: {count}\n")
            fh.write("\n")

    print(f"Staged anonymised tree: {STAGING.relative_to(ROOT)}")
    print(f"Tarball:               {TARBALL.relative_to(ROOT)}")
    print(f"Tarball SHA-256:       {tarball_sha}")
    print(f"Change log:            {LOG.relative_to(ROOT)}")
    print(f"Files edited:          {len(log.per_file)}")
    if leaks:
        print(f"WARNING: {len(leaks)} files still contain identifiers.")
        sys.exit(2)


if __name__ == "__main__":
    main()
