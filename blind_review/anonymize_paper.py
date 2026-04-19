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


ANON_AUTHOR_BLOCK = """\\author{%
  Anonymous Author(s)\\\\
  Affiliation withheld for blind review\\\\
  \\texttt{anonymous@neurips.cc}%
}"""


ANON_ACKN_BLOCK = r"""\section*{Acknowledgments}
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


def anonymize(text: str) -> tuple[str, list[str]]:
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
    assert start is not None, "author block not found"
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
        r"\\usepackage{neurips_2026} % anonymous submission mode",
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

    ack_section_pattern = re.compile(
        r"\\section\*\{Acknowledgments\}\s*\nWe thank.*?(?=\n\\bibliographystyle|\n\\bibliography|\n\\begin\{thebibliography\})",
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


def main() -> None:
    src_text = SRC.read_text(encoding="utf-8")
    anon_text, changes = anonymize(src_text)

    DST.write_text(anon_text, encoding="utf-8")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(changes) + "\n", encoding="utf-8")

    print(f"Wrote {DST.relative_to(ROOT)}")
    print(f"Change log: {LOG.relative_to(ROOT)}")
    for entry in changes:
        print(entry)


if __name__ == "__main__":
    main()
