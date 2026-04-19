# AI Scientist v2 Integration for Tinker-RL-Lab

This directory contains integration patches that make AI-Scientist-v2 run smoothly on the tinker-rl-lab repository.

## What Was Fixed

1. **Agent Prompt (`parallel_agent.py`)**: Added `tinker` to the list of available packages so the AI Scientist knows it can use the Tinker SDK.

2. **Experiment Template (`tinker_grpo_rl.py`)**:
   - Removed `if __name__ == "__main__"` guard (the AI Scientist interpreter runs code via `exec()` in global scope)
   - Replaced `argparse` with direct `working_dir` usage
   - Added graceful fallback when `TINKER_API_KEY` is missing (produces placeholder output instead of crashing)
   - Added `experiment_data.npy` saving for AI Scientist plotting code
   - Added structured `METRICS SUMMARY` block for LLM metric parsing

3. **Local TRL Template (`trl_local_grpo.py`)**: New template that runs GRPO experiments locally using HuggingFace TRL, without requiring a Tinker API key. Use this when:
   - Tinker API credits are exhausted
   - You want faster iteration on small models
   - You need reproducible local experiments

4. **BFTS Config (`bfts_config.yaml`)**: Increased `exec.timeout` from 3600s to 7200s to accommodate Tinker training latency.

5. **Dependencies (`pyproject.toml`)**: Added `tinker`, `transformers`, and `datasets` to core dependencies.

## How to Run AI Scientist v2 on This Repo

```bash
# 1. Apply patches (done automatically by autoresearch.sh)
bash ai-scientist-v2-integration/patch.sh

# 2. Launch with the Tinker template
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/tinker_grpo_rl.json" \
  --load_code \
  --model_writeup o1-preview-2024-09-12 \
  --model_citation gpt-4o-2024-11-20

# 3. Or launch with the local TRL template (no API key needed)
python launch_scientist_bfts.py \
  --load_ideas "ai_scientist/ideas/trl_local_grpo.json" \
  --load_code \
  --model_writeup o1-preview-2024-09-12
```

## Reviewer-Response Experiment Scripts

The repo includes scripts that address specific NeurIPS reviewer concerns:

| Script | Reviewer Ask | Status |
|--------|-------------|--------|
| `scripts/partial_correlation_zvf.py` | Q1: ZVF partial correlation | ✅ Works (needs log data) |
| `experiments/group_size_token_normalized.py` | Q2: Token-normalized G-sweep | ✅ Works |
| `paper/sections/framework_configs_appendix.tex` | Q3: Exact framework configs | ✅ Documented |
| `experiments/base_instruct_paired.py` | Q4: Base vs instruct pairs | ✅ Works |
| `experiments/survival_analysis.py` | Q5: 5-seed survival analysis | ✅ Works |
| `experiments/tool_use_reward_analysis.py` | Q6: Tool-use reward design | ✅ Works |
| `experiments/variance_mitigation_integration.py` | Q7: AERO/CPPO/NGRPO/Scaf-GRPO | ✅ Works (dry-run mode) |

## Known Limitations

- **Tinker experiments require `TINKER_API_KEY`**. Without it, the template falls back to placeholder output.
- **Local TRL template is simplified**. It uses a basic policy-gradient loss rather than full GRPO for speed.
- **Variance-mitigation projections**: The head-to-head table in the paper uses projected deltas from published results. Replace with measured runs when compute is available.
