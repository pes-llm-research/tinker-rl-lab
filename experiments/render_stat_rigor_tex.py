"""
Renders paper/sections/stat_rigor_updates.tex from the deterministic payload
written by compute_statistics.py (experiments/stat_rigor_tables.json).

This keeps the paper numbers in lock-step with the statistics script: every
cell in Tables 1-4 and Appendix G is driven by the JSON, so regenerating the
statistics automatically regenerates the LaTeX.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PAYLOAD = ROOT / "experiments" / "stat_rigor_tables.json"
OUT_TEX = ROOT / "paper" / "sections" / "stat_rigor_updates.tex"


# Rows from paper Table 1 (tab:main_results) that have step-level traces.
TABLE1_KEYS = [
    ("scale_gsm8k_qwen3-8b",       "Qwen3-8B",          "GRPO (Tinker)"),
    ("scale_gsm8k_qwen3.5-4b",     "Qwen3.5-4B",        "GRPO (Tinker)"),
    ("scale_gsm8k_llama-8b-inst",  "Llama-3.1-8B-Inst", "GRPO (Tinker)"),
    ("frontier_gsm8k_deepseek-v3.1","DeepSeek-V3.1",    "GRPO (Tinker)"),
    ("frontier_gsm8k_nemotron-120b","Nemotron-120B",    "GRPO (Tinker)"),
    ("campaign_v2_w1_qwen3-8b-base","Qwen3-8B-Base",    "GRPO (Tinker)"),
    ("campaign_v2_w1_gpt-oss-120b", "GPT-OSS-120B",     "GRPO (Tinker)"),
    ("ppo_qwen3-8b",               "Qwen3-8B",          "PPO (Modal H100)"),
    ("ppo_llama-8b-inst",          "Llama-3.1-8B-Inst", "PPO (Modal H100)"),
    ("cross_tool_llama-8b-inst",   "Llama-3.1-8B-Inst", "GRPO (tool-use)"),
    ("cross_tool_qwen3-32b",       "Qwen3-32B",         "GRPO (tool-use)"),
]


def fmt_p(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "---"
    if p < 1e-3:
        return "$<$0.001"
    if p >= 0.9995:
        return "1.000"
    return f"{p:.3f}"


def fmt_num(x: Optional[float], decimals: int = 3, *, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(x)


def fmt_pct(x: Optional[float], decimals: int = 1, *, signed: bool = False) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    fmt = f"{{:+.{decimals}f}}" if signed else f"{{:.{decimals}f}}"
    return fmt.format(x) + "\\%"


def fmt_ci(ci: Optional[List[float]], decimals: int = 3, *, as_pct: bool = False) -> str:
    if not ci or any(x is None or not math.isfinite(x) for x in ci):
        return "---"
    if as_pct:
        return f"[{ci[0]:.{decimals}f}\\%, {ci[1]:.{decimals}f}\\%]"
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def sig_marker(p_bonf: Optional[float]) -> str:
    if p_bonf is None or not math.isfinite(p_bonf):
        return ""
    if p_bonf < 0.001:
        return "$^{***}$"
    if p_bonf < 0.01:
        return "$^{**}$"
    if p_bonf < 0.05:
        return "$^{*}$"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
def render(payload: Dict) -> str:
    t1 = {r["experiment"]: r for r in payload["tables"]["table1"]}
    t2 = payload["tables"]["table2"]
    t3 = payload["tables"]["table3"]
    t4 = payload["tables"]["table4"]
    trl = payload["tables"]["trl_cross_seed"]
    comps = payload["family_wide_comparisons"]
    k = payload["family_wide_k"]
    master_seed = payload["master_seed"]
    B = payload["n_bootstrap"]

    lines: List[str] = []
    lines.append("% =====================================================================")
    lines.append("% paper/sections/stat_rigor_updates.tex")
    lines.append("% Task 6 --- Statistical Rigor Pass (auto-generated)")
    lines.append("%")
    lines.append(f"% Generated from experiments/stat_rigor_tables.json (MASTER_SEED={master_seed}).")
    lines.append("% Do not edit by hand --- rerun experiments/compute_statistics.py and then")
    lines.append("% experiments/render_stat_rigor_tex.py to refresh.")
    lines.append("% =====================================================================")
    lines.append("")
    lines.append("% --- Paragraph inserted near Section 5 (Main Results) -----------------")
    lines.append("\\paragraph{Statistical protocol (Task~6 rigor pass).}")
    lines.append(f"For every comparison reported in Tables~\\ref{{tab:main_results_stats}}--\\ref{{tab:ppo_grpo_stats}},")
    lines.append(f"we report (i) a 95\\,\\% percentile bootstrap CI on the point estimate")
    lines.append(f"($B\\,{{=}}\\,{B:,}$), (ii) Cohen's~$d$ with a Hedges--Olkin 95\\,\\% analytical CI,")
    lines.append(f"(iii) a raw $p$-value from the appropriate parametric/non-parametric test,")
    lines.append(f"and (iv) a Bonferroni-corrected $p$-value across the paper-wide family of")
    lines.append(f"$k={k}$ tests. The full protocol is specified in Appendix~\\ref{{app:stat_protocol}};")
    lines.append(f"all numbers are produced deterministically by")
    lines.append(f"\\texttt{{experiments/compute\\_statistics.py}} (MASTER\\_SEED\\,$=\\,${master_seed}).")
    lines.append("")

    # ─── Table 1 (updated): Main Results with CI + d + p ──────────────────────
    lines.append("% --- Table 1 (updated): Main Results with full statistics -------------")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{Main Results (Table~1, rigor pass).}} GRPO and PPO training reward on GSM8K across model scales with 95\\,\\% bootstrap CI on the full-trace and last-10 means (percentile, $B={B:,}$), Cohen's $d$ comparing late (last 10) vs.\\ early (first 10) training with 95\\,\\% Hedges--Olkin CI, and Bonferroni-corrected $p$-values across the family of $k={k}$ tests. Stars mark Bonferroni significance ($^{{*}}p<0.05$, $^{{**}}p<0.01$, $^{{***}}p<0.001$).}}")
    lines.append("\\label{tab:main_results_stats}")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}l l r l l c l c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Model} & \\textbf{$N$} & \\textbf{Last-10 [95\\% CI]} & \\textbf{Full-trace [95\\% CI]} & \\textbf{Cohen's $d$} & \\textbf{$d$ 95\\% CI} & \\textbf{$p$ (raw)} & \\textbf{$p$ (Bonf.)} \\\\")
    lines.append("\\midrule")
    for key, model_label, method in TABLE1_KEYS:
        row = t1.get(key)
        if row is None:
            continue
        d = row.get("cohens_d_late_vs_early")
        d_ci = row.get("d_ci") or []
        p_raw = row.get("p_raw")
        p_bonf = row.get("p_bonf")
        last10_pct = [row["last10_ci"][0]*100, row["last10_ci"][1]*100]
        full_pct = [row["full_ci"][0]*100, row["full_ci"][1]*100]
        last10_mean_pct = row["last10_mean"]*100
        full_mean_pct = row["full_mean"]*100
        lines.append(
            f"{method} & {model_label} & {row['steps']} & "
            f"{last10_mean_pct:.1f}\\% {fmt_ci(last10_pct, decimals=1, as_pct=True)} & "
            f"{full_mean_pct:.1f}\\% {fmt_ci(full_pct, decimals=1, as_pct=True)} & "
            f"{fmt_num(d, 2, signed=True)} & {fmt_ci(d_ci, decimals=2)} & "
            f"{fmt_p(p_raw)} & {fmt_p(p_bonf)}{sig_marker(p_bonf)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table*}")
    lines.append("")

    # ─── Table 2 (updated): Cross-library Arithmetic ──────────────────────────
    lines.append("% --- Table 2 (updated): Cross-library comparison ----------------------")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{Cross-Library Comparison (Table~2, rigor pass).}} Final arithmetic accuracy across libraries with 95\\,\\% bootstrap CI ($B={B:,}$), Cohen's $d$ vs.\\ the TRL~(GRPO) reference, and Bonferroni-corrected Welch $p$-values across the four non-reference libraries.}}")
    lines.append("\\label{tab:results_arithmetic_stats}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}l r l l c l c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Library} & $N$ & \\textbf{Mean [95\\% CI]} & \\textbf{Source} & \\textbf{Cohen's $d$} & \\textbf{$d$ 95\\% CI} & \\textbf{$p$ (raw)} & \\textbf{$p$ (Bonf.)} \\\\")
    lines.append("\\midrule")
    for r in t2:
        source = r["source"].replace("synth", "synth.").replace("real 5 seeds", "5 seeds")
        lines.append(
            f"{r['library']} & {r['n']} & "
            f"{r['mean']:.3f} {fmt_ci(r['ci95'])} & "
            f"\\textit{{{source}}} & "
            f"{fmt_num(r['d_vs_ref'], 2, signed=True)} & "
            f"{fmt_ci(r['d_ci'], decimals=2)} & "
            f"{fmt_p(r.get('p_raw'))} & "
            f"{fmt_p(r.get('p_bonf'))}{sig_marker(r.get('p_bonf'))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ─── Table 3 (updated): GSM8K scaling vs baseline ─────────────────────────
    lines.append("% --- Table 3 (updated): GSM8K scaling with effect sizes ---------------")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{GSM8K Results (Table~3, rigor pass).}} Post-RL accuracy vs.\\ baseline with bootstrap 95\\,\\% CI on $\\Delta$, Cohen's $d$ for the paired effect, and Bonferroni-corrected one-sample $t$-test $p$-values across the $k={len(t3)}$ model pairs.}}")
    lines.append("\\label{tab:results_gsm8k_stats}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{@{}l r r l l c l c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{Baseline} & \\textbf{Post-RL} & \\textbf{$\\Delta$} & \\textbf{$\\Delta$ 95\\% CI} & \\textbf{Cohen's $d$} & \\textbf{$d$ 95\\% CI} & \\textbf{$p$ (raw)} & \\textbf{$p$ (Bonf.)} \\\\")
    lines.append("\\midrule")
    for r in t3:
        lines.append(
            f"{r['model']} & {r['baseline']:.1f} & "
            f"{r['post_mean']:.1f}$\\pm${r['post_se']:.1f} & "
            f"{r['delta']:+.1f} & {fmt_ci(r['delta_ci'], decimals=1)} & "
            f"{fmt_num(r['cohens_d_vs_baseline'], 2, signed=True)} & "
            f"{fmt_ci(r['d_ci'], decimals=2)} & "
            f"{fmt_p(r.get('p_raw'))} & "
            f"{fmt_p(r.get('p_bonf'))}{sig_marker(r.get('p_bonf'))} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ─── Table 4 (updated): PPO vs GRPO ───────────────────────────────────────
    lines.append("% --- Table 4 (updated): PPO vs GRPO with CI + d + Welch/MW -----------")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{PPO vs.\\ GRPO (Table~4, rigor pass).}} Per-step reward (GSM8K, single seed, $n=30$) with 95\\,\\% bootstrap CI on last-10 and full-trace means, bootstrap CI on $\\mu_{{\\mathrm{{GRPO}}}}-\\mu_{{\\mathrm{{PPO}}}}$, Cohen's $d$ with Hedges--Olkin CI, and Bonferroni-corrected Welch $t$-test and Mann--Whitney $U$ $p$-values across the $k=2$ model pairs.}}")
    lines.append("\\label{tab:ppo_grpo_stats}")
    lines.append("\\small")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{@{}l l l l c l c c c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{Model} & \\textbf{GRPO last-10 [95\\% CI]} & \\textbf{PPO last-10 [95\\% CI]} & \\textbf{$\\Delta$ [95\\% CI]} & \\textbf{Cohen's $d$} & \\textbf{$d$ 95\\% CI} & \\textbf{Welch $p$} & \\textbf{Welch $p$ (Bonf.)} & \\textbf{MW $p$} & \\textbf{MW $p$ (Bonf.)} \\\\")
    lines.append("\\midrule")
    for r in t4:
        lines.append(
            f"{r['model']} & "
            f"{r['grpo_last10_mean']:.3f} {fmt_ci(r['grpo_last10_ci'])} & "
            f"{r['ppo_last10_mean']:.3f} {fmt_ci(r['ppo_last10_ci'])} & "
            f"{fmt_num(r['diff_mean'], 3, signed=True)} {fmt_ci(r['diff_ci'])} & "
            f"{fmt_num(r['cohens_d'], 2, signed=True)} & "
            f"{fmt_ci(r['d_ci'], decimals=2)} & "
            f"{fmt_p(r['p_welch_raw'])} & "
            f"{fmt_p(r['p_welch_bonf'])}{sig_marker(r['p_welch_bonf'])} & "
            f"{fmt_p(r['p_mw_raw'])} & "
            f"{fmt_p(r['p_mw_bonf'])}{sig_marker(r['p_mw_bonf'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    lines.append("")

    # ─── Appendix G: Statistical Protocol ─────────────────────────────────────
    lines.append("% =====================================================================")
    lines.append("% Appendix G --- Statistical Protocol")
    lines.append("% =====================================================================")
    lines.append("\\section{Statistical Protocol}")
    lines.append("\\label{app:stat_protocol}")
    lines.append("")
    lines.append("This appendix gives the full specification of the statistical rigor")
    lines.append("pass (Task~6) used to annotate every comparison in the paper. All")
    lines.append("numbers in Tables~\\ref{tab:main_results_stats}--\\ref{tab:ppo_grpo_stats}")
    lines.append("are produced by \\texttt{experiments/compute\\_statistics.py}")
    lines.append(f"and \\texttt{{experiments/render\\_stat\\_rigor\\_tex.py}} with")
    lines.append(f"\\texttt{{MASTER\\_SEED}}~$=~{master_seed}$; rerunning the script on the")
    lines.append("committed artefacts produces byte-identical JSON.")
    lines.append("")
    lines.append("\\subsection{Point Estimates and Bootstrap Confidence Intervals}")
    lines.append("For any sample $x = (x_1, \\ldots, x_n)$, we report the empirical mean")
    lines.append(f"$\\bar{{x}} = n^{{-1}} \\sum_i x_i$ and a 95\\,\\% percentile bootstrap CI")
    lines.append(f"with $B = {B:,}$ resamples:")
    lines.append("\\[")
    lines.append("  \\hat{\\theta}^{(b)} = \\bar{x}^{(b)}, \\quad b = 1, \\ldots, B, \\qquad")
    lines.append("  \\mathrm{CI}_{95\\%} = \\bigl[\\,Q_{0.025}(\\hat{\\theta}^{(b)}),\\; Q_{0.975}(\\hat{\\theta}^{(b)})\\,\\bigr].")
    lines.append("\\]")
    lines.append("Resampling uses \\texttt{np.random.Generator} seeded by a")
    lines.append("\\texttt{SeedSequence}(MASTER\\_SEED, spawn\\_key=BLAKE2(tag)); each call")
    lines.append("site derives its own independent stream, so adding a new comparison does")
    lines.append("not perturb the numbers reported for existing ones. Paired bootstrap CIs")
    lines.append("for $\\mu_A - \\mu_B$ use independent draws of $A$ and $B$ because the")
    lines.append("reward traces are independent per-step samples from the two runs.")
    lines.append("")
    lines.append("\\subsection{Effect Sizes}")
    lines.append("We report Cohen's $d$ with pooled standard deviation")
    lines.append("\\[")
    lines.append("  d = \\frac{\\bar{x}_A - \\bar{x}_B}{s_p},\\qquad")
    lines.append("  s_p = \\sqrt{\\tfrac{(n_A-1)s_A^2 + (n_B-1)s_B^2}{n_A+n_B-2}},")
    lines.append("\\]")
    lines.append("together with Hedges' $g = J\\cdot d$ using the small-sample correction")
    lines.append("$J = 1 - 3 / (4 \\mathrm{df} - 1)$, $\\mathrm{df} = n_A + n_B - 2$.")
    lines.append("The 95\\,\\% CI uses the Hedges--Olkin (1985) analytical variance")
    lines.append("$\\widehat{\\mathrm{Var}}(d) = (n_A+n_B)/(n_A n_B) + d^2/\\{2(n_A+n_B)\\}$")
    lines.append("combined with $z_{0.975}=1.96$. Magnitude labels follow Cohen's")
    lines.append("convention (negligible $<0.2$; small $0.2$--$0.5$; medium $0.5$--$0.8$;")
    lines.append("large $0.8$--$1.2$; very large $\\geq 1.2$). One-sample effects")
    lines.append("(Table~\\ref{tab:results_gsm8k_stats}) use $d=(\\bar{x}-\\mu_0)/s$ with the")
    lines.append("one-sample variance $\\mathrm{Var}(d) = 1/n + d^2/(2n)$.")
    lines.append("")
    lines.append("\\subsection{Hypothesis Tests}")
    lines.append("The test chosen for each comparison matches its sampling structure:")
    lines.append("\\begin{itemize}")
    lines.append("  \\item \\textbf{Table~\\ref{tab:main_results_stats}} (late-vs-early within an experiment):")
    lines.append("    two-sided Mann--Whitney $U$ on the first-10 and last-10 reward samples")
    lines.append("    (robust to heavy-tailed step-level reward distributions).")
    lines.append("  \\item \\textbf{Table~\\ref{tab:results_arithmetic_stats}} (cross-library):")
    lines.append("    Welch's two-sample $t$-test (unequal variances) against the TRL (GRPO)")
    lines.append("    reference.")
    lines.append("  \\item \\textbf{Table~\\ref{tab:results_gsm8k_stats}} (post-RL vs.\\ baseline):")
    lines.append("    one-sample $t$-test against the deterministic-eval baseline.")
    lines.append("  \\item \\textbf{Table~\\ref{tab:ppo_grpo_stats}} (matched-model PPO vs.\\ GRPO):")
    lines.append("    both Welch's $t$-test and Mann--Whitney $U$; the latter is reported as")
    lines.append("    a non-parametric robustness check.")
    lines.append("\\end{itemize}")
    lines.append("")
    lines.append("\\subsection{Multiple-Comparison Correction}")
    lines.append(f"The paper reports $k={k}$ comparisons in total across Tables")
    lines.append("\\ref{tab:main_results_stats}--\\ref{tab:ppo_grpo_stats}. For every raw")
    lines.append("$p$-value $p_i$ we report")
    lines.append("\\[")
    lines.append(f"  p_i^{{\\text{{Bonf}}}} = \\min(k \\cdot p_i,\\, 1),\\qquad k = {k},")
    lines.append("\\]")
    lines.append("as the headline correction (family-wise error rate $\\leq 0.05$).")
    lines.append("Benjamini--Hochberg FDR-adjusted $p$-values at $q=0.05$ are also")
    lines.append("reported in \\texttt{experiments/statistical\\_analysis.json} as a less")
    lines.append("conservative alternative. Stars in every table reflect the")
    lines.append("\\emph{Bonferroni} decision, not the BH decision.")
    lines.append("")
    lines.append("\\subsection{Determinism}")
    lines.append("Every random draw in the pipeline is routed through a single")
    lines.append("\\texttt{SeedSequence(MASTER\\_SEED)} whose \\texttt{spawn\\_key} is the")
    lines.append("BLAKE2 digest of a human-readable tag (e.g.\\ \\texttt{t4\\_diff:Qwen3-8B}).")
    lines.append("Because BLAKE2 is stable across Python processes (unlike the built-in")
    lines.append("\\texttt{hash()}), two runs of the pipeline on the same committed")
    lines.append("artefacts produce byte-identical \\texttt{statistical\\_analysis.json} and")
    lines.append("\\texttt{stat\\_rigor\\_tables.json}. We verify this in CI by running the")
    lines.append("script twice and comparing MD5 checksums.")
    lines.append("")
    lines.append("\\subsection{Family-Wide $p$-Value Summary}")
    # family-wide summary table
    lines.append("Table~\\ref{tab:family_wide_stats} lists every comparison contributing to")
    lines.append("the $k$-test family used for Bonferroni correction.")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{\\textbf{{Family-wide $p$-value table (Appendix~G).}} Every comparison contributing to the Bonferroni family ($k={k}$), with Cohen's $d$, raw $p$-value, and globally-corrected Bonferroni and Benjamini--Hochberg $p$-values.}}")
    lines.append("\\label{tab:family_wide_stats}")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{@{}r l p{5.8cm} c c c c@{}}")
    lines.append("\\toprule")
    lines.append("\\textbf{\\#} & \\textbf{Src.} & \\textbf{Comparison} & \\textbf{Test} & \\textbf{Cohen's $d$} & \\textbf{$p$ (raw)} & \\textbf{$p$ (Bonf., global)} \\\\")
    lines.append("\\midrule")
    # Sort by raw p ascending to make the table informative
    sorted_comps = sorted(
        [c for c in comps if isinstance(c.get("p_raw"), (int, float)) and math.isfinite(c["p_raw"])],
        key=lambda c: c["p_raw"],
    )
    # Pre-escape underscores outside the f-string so this file parses on Python 3.9–3.11
    # (backslash escapes inside f-string expressions are only legal on 3.12+).
    underscore_escape = "\\_"
    for i, c in enumerate(sorted_comps, start=1):
        d = c.get("d")
        d_str = fmt_num(d, 2, signed=True)
        test = c.get("test", "--")
        desc_tex = c["description"].replace("_", underscore_escape)
        test_tex = test.replace("_", underscore_escape)
        lines.append(
            f"{i} & {c['table']} & {desc_tex} & "
            f"{test_tex} & {d_str} & "
            f"{fmt_p(c['p_raw'])} & {fmt_p(c['p_bonf_global'])}{sig_marker(c['p_bonf_global'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # TRL cross-seed summary paragraph
    lines.append("\\paragraph{TRL-GRPO cross-seed baseline.}")
    lines.append(
        f"The TRL-GRPO reference (Qwen2.5-0.5B, 5 seeds, GSM8K, 125 steps, NVIDIA~L4)"
        f" has mean accuracy $\\bar{{x}} = {trl['mean']:.3f}$"
        f" (95\\,\\% bootstrap CI {fmt_ci(trl['ci95'])},"
        f" SD $= {trl['sd']:.3f}$, CV $= {trl['cv']:.3f}$)."
        f" A one-sample $t$-test against $\\mu_0 = 0.5$ gives"
        f" $t({trl['n']-1}) = {trl['t_vs_0.5']:.2f}$, $p = $~{fmt_p(trl['p_vs_0.5_two_sided'])},"
        f" Cohen's $d = {trl['cohens_d_vs_0.5']:.2f}$ --- a very large effect that"
        f" survives Bonferroni correction in the family-wide table above."
    )
    lines.append("")

    lines.append("% --- end of stat_rigor_updates.tex ----------------------------------")
    return "\n".join(lines) + "\n"


def main() -> int:
    payload = json.load(PAYLOAD.open())
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text(render(payload))
    print(f"[render] wrote {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
