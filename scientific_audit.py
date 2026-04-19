#!/usr/bin/env python3
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER_TEX = ROOT / "reports/final/grpo_agentic_llm_paper.tex"
PAPER_TEX_ANON = ROOT / "reports/final/grpo_agentic_llm_paper_anonymous.tex"
PAPER_MD = ROOT / "reports/final/grpo_agentic_llm_paper.md"
REPORT_MD = ROOT / "reports/final/capstone_final_report.md"
SUBMISSION_CHECKLIST = ROOT / "reports/final/SUBMISSION_CHECKLIST.md"
EVAL_PY = ROOT / "reports/final/evaluate_gsm8k_test.py"
FINAL_DIR = ROOT / "reports/final"
RESULT_JSONS = [FINAL_DIR / "gsm8k_base_results.json", FINAL_DIR / "gsm8k_test_results.json"]

issues = []


def add(path: Path, code: str, message: str):
    issues.append((str(path.relative_to(ROOT)), code, message))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check_paper():
    tex = read(PAPER_TEX)
    anon = read(PAPER_TEX_ANON)
    paper_md = read(PAPER_MD)
    md = read(REPORT_MD)
    checklist = read(SUBMISSION_CHECKLIST)
    supp = read(FINAL_DIR / "supplementary_appendix.tex")

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if abstract_match:
        abstract = abstract_match.group(1).lower()
        if (
            "held-out" not in abstract
            and "training-set" not in abstract
            and "evaluation scope" not in abstract
        ):
            add(
                PAPER_TEX,
                "paper.abstract.scope",
                "LaTeX abstract reports GSM8K gains without explicitly saying they are training-set reward metrics, risking overclaim.",
            )

    if "held-out" not in tex.lower() and "training-set reward" not in tex.lower():
        add(
            PAPER_TEX,
            "paper.global.scope",
            "LaTeX paper lacks an explicit held-out-vs-training-set evaluation scope warning.",
        )

    if "publishable confidence intervals" in md.lower():
        add(
            REPORT_MD,
            "report.overclaim.publishable",
            "Capstone report claims 'publishable confidence intervals' despite n=3 seeds and no held-out evaluation.",
        )

    anon_abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", anon, re.S)
    if anon_abstract_match:
        anon_abstract = anon_abstract_match.group(1).lower()
        if (
            "held-out" not in anon_abstract
            and "training-set" not in anon_abstract
            and "evaluation scope" not in anon_abstract
        ):
            add(
                PAPER_TEX_ANON,
                "paper_anon.abstract.scope",
                "Anonymous LaTeX abstract still reports GSM8K gains without an explicit training-set-vs-held-out caveat.",
            )

    if "held-out" not in paper_md.lower() and "training-set reward" not in paper_md.lower():
        add(
            PAPER_MD,
            "paper_md.global.scope",
            "Markdown paper lacks an explicit held-out-vs-training-set scope warning.",
        )

    if re.search(r"\|\s*GSM8K\s*\|\s*30\.0% \± 2\.5% \(3 seeds\)\s*\|", checklist):
        add(
            SUBMISSION_CHECKLIST,
            "checklist.gsm8k.label",
            "Submission checklist labels GSM8K as a generic result instead of explicitly marking it as training-set reward.",
        )

    for path, text in [
        (PAPER_TEX, tex),
        (PAPER_TEX_ANON, anon),
        (PAPER_MD, paper_md),
    ]:
        if "confirms grpo training stability" in text.lower():
            add(
                path,
                "paper.stability.overclaim",
                "Paper claims the 3-seed GSM8K result 'confirms' training stability; this should be softened to a more accurate characterization.",
            )

    for path, text in [
        (REPORT_MD, md),
        (PAPER_MD, paper_md),
        (FINAL_DIR / "supplementary_appendix.tex", supp),
    ]:
        low = text.lower()
        if "99% gsm8k accuracy" in low or "99\\% gsm8k accuracy" in low:
            add(
                path,
                "gsm8k.peak_accuracy.overclaim",
                "A 99% GSM8K statement appears without making clear that it refers to a peak training-step metric rather than held-out benchmark accuracy.",
            )


def _is_name(node, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _const_str(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def check_eval_script():
    tree = ast.parse(read(EVAL_PY))
    source = read(EVAL_PY)

    parser_has_seed = "--seed" in source
    parser_has_split = "--split" in source
    split_choices_locked = 'choices=["test"]' in source or "choices=['test']" in source
    checkpoint_arg_used = False
    fallback_last_number = "Fallback: extract last number" in source
    do_sample_true = False
    default_temp_nonzero = False
    has_dataset_split_metadata = '"dataset_split": args.split' in source
    has_do_sample_metadata = '"do_sample": args.do_sample' in source
    has_seed_metadata = '"seed": args.seed' in source
    has_model_source_metadata = '"model_source":' in source

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "add_argument":
                for arg in node.args:
                    s = _const_str(arg)
                    if s == "--temperature":
                        for kw in node.keywords:
                            if (
                                kw.arg == "default"
                                and isinstance(kw.value, ast.Constant)
                                and kw.value.value not in (0, 0.0)
                            ):
                                default_temp_nonzero = True
                    if s == "--seed":
                        parser_has_seed = True
            if node.func.attr == "generate":
                for kw in node.keywords:
                    if (
                        kw.arg == "do_sample"
                        and isinstance(kw.value, ast.Constant)
                        and kw.value.value is True
                    ):
                        do_sample_true = True
        if isinstance(node, ast.Name) and node.id == "checkpoint_path":
            checkpoint_arg_used = True

    if default_temp_nonzero:
        add(
            EVAL_PY,
            "eval.nondeterministic.default_temp",
            "Evaluation defaults to temperature=0.7, which makes headline accuracy nondeterministic.",
        )
    if do_sample_true:
        add(
            EVAL_PY,
            "eval.nondeterministic_sampling",
            "HF evaluation uses do_sample=True instead of deterministic decoding, weakening rigor and reproducibility.",
        )
    if not parser_has_seed:
        add(
            EVAL_PY,
            "eval.missing_seed",
            "Evaluation script has no seed control for stochastic generation.",
        )
    if not parser_has_split:
        add(
            EVAL_PY,
            "eval.missing_split_arg",
            "Evaluation script does not record which dataset split it evaluates.",
        )
    if parser_has_split and not split_choices_locked:
        add(
            EVAL_PY,
            "eval.unlocked_split",
            "Evaluation script allows non-test splits; held-out evaluation should be locked to the GSM8K test split to avoid accidental train-set reporting.",
        )
    if not checkpoint_arg_used:
        add(
            EVAL_PY,
            "eval.unused_checkpoint_path",
            "--checkpoint_path is declared but never used, so local checkpoint evaluation is broken/misleading.",
        )
    if fallback_last_number:
        add(
            EVAL_PY,
            "eval.lenient_answer_extraction",
            "Answer extraction falls back to the last number in the response, which can overcount correctness and invite benchmark leakage.",
        )
    if not has_dataset_split_metadata:
        add(
            EVAL_PY,
            "eval.missing_split_metadata",
            "Saved evaluation results do not record the dataset split, weakening auditability.",
        )
    if not has_do_sample_metadata:
        add(
            EVAL_PY,
            "eval.missing_sampling_metadata",
            "Saved evaluation results do not record whether decoding was greedy or sampled.",
        )
    if not has_seed_metadata:
        add(
            EVAL_PY,
            "eval.missing_seed_metadata",
            "Saved evaluation results do not record the evaluation seed.",
        )
    if not has_model_source_metadata:
        add(
            EVAL_PY,
            "eval.missing_model_source_metadata",
            "Saved evaluation results do not record the exact checkpoint/model source used for evaluation.",
        )


def _run(cmd, cwd: Path):
    return subprocess.run(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )


def check_latex_builds():
    cleanup = [
        "grpo_agentic_llm_paper.aux",
        "grpo_agentic_llm_paper.bbl",
        "grpo_agentic_llm_paper.blg",
        "grpo_agentic_llm_paper.log",
        "grpo_agentic_llm_paper.out",
        "grpo_agentic_llm_paper.pdf",
        "grpo_agentic_llm_paper_anonymous.aux",
        "grpo_agentic_llm_paper_anonymous.log",
        "grpo_agentic_llm_paper_anonymous.out",
        "grpo_agentic_llm_paper_anonymous.pdf",
        "supplementary_appendix.aux",
        "supplementary_appendix.log",
        "supplementary_appendix.out",
        "supplementary_appendix.pdf",
    ]
    for name in cleanup:
        path = FINAL_DIR / name
        if path.exists():
            path.unlink()

    steps = [
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "grpo_agentic_llm_paper.tex",
            ],
            "latex.main.pass1",
        ),
        (["bibtex", "grpo_agentic_llm_paper"], "latex.main.bibtex"),
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "grpo_agentic_llm_paper.tex",
            ],
            "latex.main.pass2",
        ),
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "grpo_agentic_llm_paper.tex",
            ],
            "latex.main.pass3",
        ),
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "grpo_agentic_llm_paper_anonymous.tex",
            ],
            "latex.anonymous",
        ),
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "supplementary_appendix.tex",
            ],
            "latex.supplementary.pass1",
        ),
        (
            [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                "supplementary_appendix.tex",
            ],
            "latex.supplementary.pass2",
        ),
    ]

    try:
        for cmd, code in steps:
            result = _run(cmd, FINAL_DIR)
            if result.returncode != 0:
                add(FINAL_DIR / cmd[-1], code, f"LaTeX build step failed: {' '.join(cmd)}")
                break
            if code == "latex.main.bibtex" and "Warning--empty journal" in result.stdout:
                add(
                    FINAL_DIR / "references.bib",
                    "latex.bibtex.empty_journal",
                    "BibTeX emitted 'empty journal' warnings for cited references, so the bibliography metadata is incomplete.",
                )
    finally:
        for name in cleanup:
            path = FINAL_DIR / name
            if path.exists():
                path.unlink()


def check_result_jsons():
    required_config = {
        "model",
        "model_source",
        "dataset",
        "dataset_config",
        "dataset_split",
        "n_samples",
        "temperature",
        "do_sample",
        "seed",
        "max_tokens",
        "test_size",
    }
    required_summary = {
        "correct",
        "incorrect",
        "errors",
        "attempted",
        "accuracy",
        "accuracy_percent",
    }

    for path in RESULT_JSONS:
        data = json.loads(read(path))
        if data.get("schema_version") != 2:
            add(
                path,
                "results.schema_version",
                "Result JSON does not declare the current schema_version=2.",
            )
        if data.get("evaluation_status") not in {"completed", "failed"}:
            add(
                path,
                "results.status",
                "Result JSON must declare evaluation_status as 'completed' or 'failed'.",
            )
        config = data.get("config")
        summary = data.get("summary")
        if not isinstance(config, dict) or not required_config.issubset(config):
            add(
                path,
                "results.config",
                "Result JSON is missing required evaluation provenance fields in config.",
            )
            continue
        if not isinstance(summary, dict) or not required_summary.issubset(summary):
            add(path, "results.summary", "Result JSON is missing required summary fields.")
            continue
        if config.get("dataset_split") != "test":
            add(
                path,
                "results.non_test_split",
                "Bundled GSM8K result JSON must be explicitly marked as test-split evaluation.",
            )
        attempted = summary.get("attempted")
        if attempted != summary.get("correct", 0) + summary.get("incorrect", 0):
            add(
                path,
                "results.attempted_mismatch",
                "attempted must equal correct + incorrect in the result summary.",
            )
        if data.get("evaluation_status") == "failed" and not data.get("failure_reason"):
            add(path, "results.failure_reason", "Failed evaluations must record a failure_reason.")
        if data.get("evaluation_status") == "completed" and attempted == 0:
            add(
                path,
                "results.completed_zero_attempts",
                "Completed evaluations must have at least one attempted example.",
            )


def main():
    check_paper()
    check_eval_script()
    check_latex_builds()
    check_result_jsons()
    print(f"METRIC audit_issues={len(issues)}")
    for path, code, message in issues:
        print(f"ISSUE {code} {path}: {message}")


if __name__ == "__main__":
    main()
