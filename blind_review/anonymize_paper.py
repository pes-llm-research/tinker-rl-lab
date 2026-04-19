#!/usr/bin/env python3
"""Anonymize paper/main.tex into paper/main_anon.tex for NeurIPS blind review.

Changes, in order:
  1. Replace the ``\\author{...}`` block with an anonymous block
     and remove email addresses / PES University affiliations.
  2. Switch the NeurIPS style flag from preprint to anonymised submission.
  3. Replace identifying URLs (arvindcr4, pes-llm-research, wandb entity)
     with anonymous.4open.science / anonymous placeholders.
  4. Replace HuggingFace repo paths tied to team members.
  5. Rewrite the Reproducibility ``Team Model Checkpoints'' list with
     anonymous stubs (authors names/handles removed).
  6. Rewrite the ``Acknowledgments'' and funding/compute mentions so no
     institution, project guide, or supervisor is named.
  7. Replace any remaining ``PES University'' / ``Great Learning'' strings
     with ``Anonymous Institution''.
  8. Remove header comment author guidance (historical note).

The script is idempotent: running it twice produces the same output.
Run from the repository root: ``python blind_review/anonymize_paper.py``.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "paper" / "main.tex"
DST = ROOT / "paper" / "main_anon.tex"
LOG = ROOT / "blind_review" / "paper_changes.log"

# Supplementary \input{} targets that also contain identifying information and
# need parallel anonymized copies.  Each entry is (src, dst) relative to the
# repository root.  The rewrite pass below also rewrites any \input{SRC}
# references inside main_anon.tex to point at the DST copies.
INCLUDE_FILES = [
    ("paper/ethics_statement.tex",       "paper/ethics_statement_anon.tex"),
    ("paper/sections/abstract.tex",      "paper/sections/abstract_anon.tex"),
    ("paper/sections/intro.tex",         "paper/sections/intro_anon.tex"),
    ("paper/sections/related_work_v2.tex", "paper/sections/related_work_v2_anon.tex"),
    ("paper/sections/conclusion.tex",    "paper/sections/conclusion_anon.tex"),
    ("paper/sections/stat_rigor_updates.tex", "paper/sections/stat_rigor_updates_anon.tex"),
    ("paper/sections/checklist.tex",     "paper/sections/checklist_anon.tex"),
]


ANON_AUTHOR_BLOCK = """\\author{%
  Anonymous Author(s)\\\\
  Affiliation withheld for blind review\\\\
  \\texttt{anonymous@neurips.cc}%
}"""


ANON_ACKN_BLOCK = r"""\section*{Acknowledgments}
\label{sec:acknowledgments}
Acknowledgments are withheld for blind review. We thank the maintainers of the
open-source libraries on which this work builds (TRL, Stable Baselines3,
CleanRL, Tianshou, OpenRLHF, veRL, and the broader Hugging Face ecosystem),
the \tinker{} team for API access, the Modal team for GPU compute credits,
and Weights \& Biases for experiment-tracking infrastructure. All individual
mentors, funders, and institutional affiliations are redacted pending
acceptance.
"""


ANON_ACK_ENV_BLOCK = r"""\begin{ack}
Acknowledgments are withheld for blind review. We thank the Tinker team for
API access, the Modal team for GPU compute credits (H100 cluster), Weights
\& Biases for experiment tracking infrastructure, and Hugging Face for model
hosting. We are grateful to the open-source communities behind TRL, Stable
Baselines3, CleanRL, Tianshou, OpenRLHF, veRL, and the broader Hugging Face
ecosystem. Institutional support statements are redacted for blind review.
\end{ack}"""


TEAM_CHECKPOINTS_ANON = r"""\paragraph{Anonymous Model Checkpoints.}
Task-specific models from Section~\ref{sec:task_grpo} are released via the
anonymous mirror. Repository handles are redacted for blind review; reviewers
can download the artefacts through the anonymous 4open.science repository
link provided above. Model-card slugs in the anonymous release use the
pattern \texttt{anonymous/tinker-rl-bench-<task>-<model>}.
"""


def _record(changes, label, count):
    changes.append(f"- {label}: {count} replacement(s)")


def anonymize(text: str, *, require_author_block: bool = True) -> tuple[str, list[str]]:
    """Anonymize ``text``.

    Parameters
    ----------
    text:
        LaTeX source to process.
    require_author_block:
        ``True`` (default) for root documents where the real ``\\author{...}``
        block must be present and rewritten.  ``False`` for included fragments
        (abstract, ethics, checklist, etc.) that do not contain an author
        block; in that mode step 1 is skipped but every other substitution
        still runs.
    """
    changes: list[str] = []

    # 1. Replace the real \author{...} block. We locate it by scanning for
    # an ``\author{`` that is at the start of a line (not inside a comment)
    # and contains ``Arvind'' (first named author).
    lines = text.splitlines(keepends=True)
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("\\author{"):
            start = idx
            break
    if start is None:
        if require_author_block:
            raise AssertionError("author block not found")
        # included fragment: skip step 1 entirely and jump to the substitutions
        return _run_substitutions(text, changes, skip_author_block=True)
    depth = 0
    end = None
    for idx in range(start, len(lines)):
        for ch in lines[idx]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end is not None:
            break
    assert end is not None, "unterminated author block"
    new_lines = lines[:start] + [ANON_AUTHOR_BLOCK + "\n"] + lines[end + 1 :]
    text = "".join(new_lines)
    _record(changes, "author block replaced with Anonymous Author(s)", 1)

    # 2. Submission style flag: switch to anonymous / line-numbered mode.
    new_text, n = re.subn(
        r"\\usepackage\[preprint,nonatbib\]\{neurips_2026\}[^\n]*",
        r"\\usepackage[nonatbib]{neurips_2026} % anonymous submission mode",  # keep nonatbib so natbib is loaded exactly once
        text,
    )
    _record(changes, "neurips style flag switched to anonymous submission", n)
    text = new_text

    # 3. Author-block submission guidance (header comments lines 7--10 and the
    # inline ``For submission, use anonymous authors:'' block).
    for old, tag in [
        (
            r"% SUBMISSION NOTE:.*?https://anonymous\.4open\.science/r/tinker-rl-lab\}\n\n",
            "header author-guidance comment removed",
        ),
        (
            r"% For submission, use anonymous authors:\n% \\author\{Anonymous Authors\}\n\n",
            "inline ``for submission'' comment removed",
        ),
    ]:
        new_text, n = re.subn(old, "", text, flags=re.DOTALL)
        _record(changes, tag, n)
        text = new_text

    # 4. Header title comment: drop project-identifying tag.
    new_text, n = re.subn(
        r"^% NeurIPS 2026 Paper: TinkerRL Lab$",
        r"% NeurIPS 2026 Paper",
        text,
        flags=re.MULTILINE,
    )
    _record(changes, "header ``TinkerRL Lab'' project tag removed", n)
    text = new_text

    # 5. GitHub URLs.
    for old, new, tag in [
        (
            r"https://github\.com/arvindcr4/tinker-rl-lab",
            r"https://anonymous.4open.science/r/tinker-rl-lab",
            "github.com/arvindcr4/tinker-rl-lab -> anonymous.4open.science",
        ),
        (
            r"https://github\.com/pes-llm-research/tinker-rl-lab",
            r"https://anonymous.4open.science/r/tinker-rl-lab",
            "github.com/pes-llm-research/tinker-rl-lab -> anonymous.4open.science",
        ),
        (
            r"https://github\.com/arvindcr4/([\w-]+)",
            r"https://anonymous.4open.science/r/\1",
            "github.com/arvindcr4/<repo> -> anonymous.4open.science",
        ),
        (
            r"https://github\.com/pes-llm-research/([\w-]+)",
            r"https://anonymous.4open.science/r/\1",
            "github.com/pes-llm-research/<repo> -> anonymous.4open.science",
        ),
        (
            r"https://github\.com/madhukumara1993/qwen3-grpo",
            r"https://anonymous.4open.science/r/code-grpo",
            "github.com/madhukumara1993/qwen3-grpo redacted",
        ),
    ]:
        new_text, n = re.subn(old, new, text)
        _record(changes, tag, n)
        text = new_text

    # 6. HuggingFace URLs and repo paths.
    for old, new, tag in [
        (
            r"https://huggingface\.co/arvindcr4/tinker-rl-bench-\*",
            r"https://huggingface.co/anonymous/tinker-rl-bench-*",
            "HuggingFace arvindcr4/tinker-rl-bench-* -> anonymous/",
        ),
        (
            r"huggingface\.co/arvindcr4/",
            r"huggingface.co/anonymous/",
            "HuggingFace arvindcr4/ -> anonymous/",
        ),
        (
            r"huggingface\.co/pes-llm-research/",
            r"huggingface.co/anonymous/",
            "HuggingFace pes-llm-research/ -> anonymous/",
        ),
        (
            r"huggingface\.co/Madhu2133",
            r"huggingface.co/anonymous",
            "HuggingFace Madhu2133 -> anonymous",
        ),
        (
            r"huggingface\.co/MohammadRafiML",
            r"huggingface.co/anonymous",
            "HuggingFace MohammadRafiML -> anonymous",
        ),
        (
            r"huggingface\.co/Balasandhya",
            r"huggingface.co/anonymous",
            "HuggingFace Balasandhya -> anonymous",
        ),
        (
            r"huggingface\.co/dhruvanmurthy",
            r"huggingface.co/anonymous",
            "HuggingFace dhruvanmurthy -> anonymous",
        ),
    ]:
        new_text, n = re.subn(old, new, text)
        _record(changes, tag, n)
        text = new_text

    # 7. Weights & Biases URLs and entity names.
    for old, new, tag in [
        (
            r"https://wandb\.ai/arvindcr4-pes-university/",
            r"https://wandb.ai/anonymous/",
            "wandb entity arvindcr4-pes-university -> anonymous",
        ),
        (
            r"https://wandb\.ai/tinker-rl-lab/tinker-rl-lab-world-class",
            r"https://wandb.ai/anonymous/tinker-rl-bench",
            "wandb project URL -> anonymous mirror",
        ),
        (
            r"project:\s*\\texttt\{tinker-rl-lab-world-class\}",
            r"project: \\texttt{tinker-rl-bench}",
            "wandb project name ``tinker-rl-lab-world-class'' -> ``tinker-rl-bench''",
        ),
        (
            r"tinker-rl-lab-world-class",
            r"tinker-rl-bench",
            "wandb project name (bare) -> tinker-rl-bench",
        ),
    ]:
        new_text, n = re.subn(old, new, text)
        _record(changes, tag, n)
        text = new_text

    # 8. Team Model Checkpoints paragraph in section sec:reproducibility.
    team_pattern = re.compile(
        r"\\paragraph\{Team Model Checkpoints\.\}.*?\\end\{itemize\}\n",
        re.DOTALL,
    )
    new_text, n = team_pattern.subn(lambda _m: TEAM_CHECKPOINTS_ANON, text)
    _record(changes, "``Team Model Checkpoints'' block replaced with anonymous stub", n)
    text = new_text

    # 9. Acknowledgments environments.
    ack_pattern = re.compile(r"\\begin\{ack\}.*?\\end\{ack\}", re.DOTALL)
    new_text, n = ack_pattern.subn(lambda _m: ANON_ACK_ENV_BLOCK, text)
    _record(changes, "\\begin{ack}..\\end{ack} environment anonymised", n)
    text = new_text

    # Match the \section*{Acknowledgments} block.  The main.tex version now
    # carries a \label{sec:acknowledgments} line between the heading and the
    # first paragraph, so the previous pattern (which required `We thank` to
    # start the very next line) no longer fired.  Tolerate any number of
    # intervening whitespace/label lines before the real body.
    ack_section_pattern = re.compile(
        r"\\section\*\{Acknowledgments\}(?:\s*\\label\{[^}]*\})?\s*\nWe thank.*?(?=\n\\bibliographystyle|\n\\bibliography|\n\\begin\{thebibliography\})",
        re.DOTALL,
    )
    new_text, n = ack_section_pattern.subn(lambda _m: ANON_ACKN_BLOCK + "\n", text)
    _record(changes, "\\section*{Acknowledgments} block anonymised", n)
    text = new_text

    # 10. Remaining institutional references anywhere in the body.
    for old, new, tag in [
        (
            r"PES University's LLM Research Group",
            r"our host institution's research group",
            "``PES University's LLM Research Group'' redacted",
        ),
        (
            r"PES University/Great Learning",
            r"Anonymous Institution",
            "``PES University/Great Learning'' redacted",
        ),
        (
            r"Great Learning / PES University",
            r"Anonymous Institution",
            "``Great Learning / PES University'' redacted",
        ),
        (r"PES University", r"Anonymous Institution", "``PES University'' redacted"),
        (r"Great Learning", r"Anonymous Institution", "``Great Learning'' redacted"),
        (
            r"Northwestern University / Anonymous Institution",
            r"Anonymous Institution",
            "``Northwestern University / ...'' redacted",
        ),
        (
            r"Northwestern University",
            r"Anonymous Institution",
            "``Northwestern University'' redacted",
        ),
        # The ethics statement refers to the shared A100 node as ``PES A100''.
        # After redacting ``PES University'' we still need to strip the node name.
        (r"PES A100", r"Institutional A100", "``PES A100'' node name redacted"),
        # ``by collaborator <Name>'' / ``by <Surname>'' in ethics data-statement
        # paragraphs must not reveal any team-member name.
        (
            r"by collaborator Mohammad Rafi",
            r"by an anonymous collaborator",
            "collaborator Mohammad Rafi redacted",
        ),
        (
            r"by collaborator Madhu\b",
            r"by an anonymous collaborator",
            "collaborator Madhu redacted",
        ),
        (
            r"by collaborator Sandhya\b",
            r"by an anonymous collaborator",
            "collaborator Sandhya redacted",
        ),
        (
            r"by collaborator Arumugam\b",
            r"by an anonymous collaborator",
            "collaborator Arumugam redacted",
        ),
        (
            r"by collaborator Dhruva\b",
            r"by an anonymous collaborator",
            "collaborator Dhruva redacted",
        ),
        # Residual mentions such as ``reported by Rafi'' / ``by Madhu''.
        (
            r"reported by Rafi\b",
            r"reported by that collaborator",
            "``reported by Rafi'' redacted",
        ),
        (
            r"\bby (Rafi|Madhu|Sandhya|Arumugam|Dhruva)\b",
            r"by that collaborator",
            "bare ``by <surname>'' redacted",
        ),
        # The ``arvindcr4/tinker-rl-bench-*'' pattern still appears as a
        # plain-text example in ethics_statement.tex.
        (
            r"arvindcr4/tinker-rl-bench-\*",
            r"anonymous/tinker-rl-bench-*",
            "``arvindcr4/tinker-rl-bench-*'' placeholder redacted",
        ),
        (
            r"pes-llm-research/tinker-rl-lab",
            r"anonymous-org/tinker-rl-lab",
            "``pes-llm-research/tinker-rl-lab'' placeholder redacted",
        ),
        (
            r"arvindcr4/tinker-rl-lab",
            r"anonymous-mirror/tinker-rl-lab",
            "``arvindcr4/tinker-rl-lab'' mirror placeholder redacted",
        ),
    ]:
        new_text, n = re.subn(old, new, text)
        _record(changes, tag, n)
        text = new_text

    # 11. Team-member surnames / handles that may appear in prose.
    for old, new, tag in [
        (r"Balasandhya/[\w.-]+", "anonymous/multiturn-toolcall", "Balasandhya/* HF slug redacted"),
        (r"Madhu2133/[\w.-]+", "anonymous/code-grpo", "Madhu2133/* HF slug redacted"),
        (r"MohammadRafiML", "anonymous", "MohammadRafiML handle redacted"),
        (
            r"dhruvanmurthy/[\w\\\\_.-]+",
            "anonymous/efficiency-grpo",
            "dhruvanmurthy/* HF slug redacted",
        ),
        (r"madhukumara1993", "anonymous", "madhukumara1993 GitHub handle redacted"),
        (
            r"arvindcr4-pes-university",
            "anonymous",
            "arvindcr4-pes-university wandb entity (bare) redacted",
        ),
        (r"arvindcr4", "anonymous", "arvindcr4 handle redacted"),
        (r"pes-llm-research", "anonymous-org", "pes-llm-research org redacted"),
    ]:
        new_text, n = re.subn(old, new, text)
        _record(changes, tag, n)
        text = new_text

    # 12. Email addresses (defence in depth).
    for old, tag in [
        (r"\\texttt\{arvindcr4@gmail\.com\}", "email arvindcr4@gmail.com"),
        (r"\\texttt\{sandhya\.jeyaraj2014@gmail\.com\}", "email sandhya.jeyaraj2014"),
        (r"\\texttt\{madhukumara1993@gmail\.com\}", "email madhukumara1993"),
        (r"\\texttt\{gmd\.rafi\.2024@gmail\.com\}", "email gmd.rafi.2024"),
        (r"\\texttt\{dhruva\.n\.murthy@gmail\.com\}", "email dhruva.n.murthy"),
        (r"\\texttt\{chettyarumugam@mail\.com\}", "email chettyarumugam"),
        (r"\\texttt\{anwesh@greatlearning\.in\}", "email anwesh@greatlearning.in"),
        (r"\\texttt\{narayana\.darapaneni@northwestern\.edu\}", "email narayana.darapaneni"),
    ]:
        new_text, n = re.subn(old, r"\\texttt{anonymous@neurips.cc}", text)
        _record(changes, f"{tag} redacted (defence in depth)", n)
        text = new_text

    return text, changes


def _run_substitutions(text: str, changes: list[str], *, skip_author_block: bool) -> tuple[str, list[str]]:
    """Internal helper used by :func:`anonymize` to apply every substitution
    pass *except* step 1 (author block rewrite).  Kept as a separate
    function so included fragments can reuse the same pipeline without
    requiring a ``\\author{}`` block."""
    # Rerun :func:`anonymize` with a sentinel that lets it skip step 1.
    # Implementation: temporarily prepend a synthetic author block, run the
    # full pipeline, then strip the synthetic block back out.  This avoids
    # duplicating the substitution list.
    sentinel_block = "\\author{%\n  Arvind C R\\\\\n  PES University\\\\\n  \\texttt{arvindcr4@gmail.com}%\n}\n"
    sentinel_tag = "%__ANON_SENTINEL_START__\n"
    synthetic = sentinel_tag + sentinel_block + "%__ANON_SENTINEL_END__\n"
    wrapped = synthetic + text
    anon_wrapped, inc_changes = anonymize(wrapped, require_author_block=True)
    # Strip anything from the top of the wrapped output down through the
    # replaced author block.  The replacement is the fixed ANON_AUTHOR_BLOCK
    # string, so locate the line that *follows* it.  We keep the sentinel
    # comments so they act as fence posts.
    marker = "%__ANON_SENTINEL_END__"
    idx = anon_wrapped.find(marker)
    if idx != -1:
        # Drop through end of that line
        newline = anon_wrapped.find("\n", idx)
        anon_wrapped = anon_wrapped[newline + 1:]
    # Extend the change log but drop the author-block line (it's synthetic)
    filtered = [c for c in inc_changes if "author block replaced" not in c]
    changes.extend(filtered)
    return anon_wrapped, changes


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    anon_text, changes = anonymize(src_text)

    # Rewrite \input{SRC} references to point at their anonymized companions.
    for src_rel, dst_rel in INCLUDE_FILES:
        src_key = Path(src_rel).relative_to("paper").with_suffix("").as_posix()
        dst_key = Path(dst_rel).relative_to("paper").with_suffix("").as_posix()
        pat = re.compile(r"\\input\{" + re.escape(src_key) + r"\}")
        new_anon_text, n = pat.subn(r"\\input{" + dst_key + r"}", anon_text)
        anon_text = new_anon_text
        changes.append(f"- rewrote \\input{{{src_key}}} -> \\input{{{dst_key}}}: {n} replacement(s)")

    DST.write_text(anon_text, encoding="utf-8")

    # Anonymize each included file the same way as main.tex. They inherit the
    # same substitution pipeline but we discard author-block / style-flag
    # changes that only matter in the root document.
    for src_rel, dst_rel in INCLUDE_FILES:
        src_path = ROOT / src_rel
        dst_path = ROOT / dst_rel
        if not src_path.exists():
            changes.append(f"- SKIPPED (missing): {src_rel}")
            continue
        inc_text = src_path.read_text(encoding="utf-8")
        anon_inc_text, inc_changes = anonymize(inc_text, require_author_block=False)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_text(anon_inc_text, encoding="utf-8")
        changes.append(f"- wrote {dst_rel} ({len(inc_changes)} passes)")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(changes) + "\n", encoding="utf-8")

    print(f"Wrote {DST.relative_to(ROOT)}")
    print(f"Change log: {LOG.relative_to(ROOT)}")
    for entry in changes:
        print(entry)


if __name__ == "__main__":
    main()
