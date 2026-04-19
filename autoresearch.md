# Autoresearch: AI-Scientist-v2 Runnability & Paper Acceptance

## Objective
1. **Phase 1**: Improve the success rate of AI-Scientist-v2 experiments when targeting the tinker-rl-lab repository.
2. **Phase 2**: Optimize the repo for NeurIPS paper acceptance by addressing reviewer concerns.

## Metrics
- **Phase 1 Primary**: `runnability_score` (0-180, higher is better) — AI Scientist infrastructure readiness
- **Phase 2 Primary**: `acceptance_score` (0-127, higher is better) — reviewer concern coverage
- **Secondary**: See benchmark outputs for full metric lists

## How to Run
- `./autoresearch.sh` — current acceptance benchmark
- `./autoresearch_runnability.sh` — runnability benchmark (archived)

## Files in Scope
- `ai-scientist-v2-integration/` — patches for AI Scientist v2
- `ai-scientist-v2-integration/ai_scientist/ideas/tinker_grpo_rl.py` — Tinker experiment template
- `ai-scientist-v2-integration/ai_scientist/ideas/trl_local_grpo.py` — Local TRL experiment template
- `ai-scientist-v2-integration/ai_scientist/treesearch/parallel_agent.py` — Agent prompt patch
- `ai-scientist-v2-integration/bfts_config.yaml` — BFTS config patch
- `pyproject.toml` — dependency declarations

## Off Limits
- Do NOT modify core AI Scientist v2 search algorithm
- Do NOT add hardcoded API keys or credentials

## Constraints
- All changes must be backwards-compatible
- Benchmark must remain fast (< 10s per run)

## What's Been Tried
### Runnability Optimizations (Score: 180/180)
- Installed missing Python deps (`tinker`, `orjson`, `datasets`) into venv
- Added `tinker` to agent's `_prompt_environment` package list
- Fixed `tinker_grpo_rl.py` template:
  - Removed `if __name__ == "__main__"` guard
  - Replaced `argparse` with direct execution
  - Added graceful `TINKER_API_KEY` fallback
  - Added `experiment_data.npy` output for plotting
  - Added structured `METRICS SUMMARY` block
- Increased BFTS `exec.timeout` from 3600s → 7200s
- Added `tinker` + `transformers` + `datasets` to `pyproject.toml` core deps

### Paper Acceptance Optimizations (Score: 127/127)
- Verified all 24 reviewer weaknesses are addressed in paper text
- Verified 7/7 critical response scripts exist and execute:
  - `scripts/partial_correlation_zvf.py`
  - `experiments/group_size_token_normalized.py`
  - `experiments/base_instruct_paired.py`
  - `experiments/survival_analysis.py`
  - `experiments/tool_use_reward_analysis.py`
  - `experiments/variance_mitigation_integration.py`
  - `paper/sections/framework_configs_appendix.tex`
- Created local TRL template (`trl_local_grpo.py`) for experiments without Tinker API
- Verified paper LaTeX builds successfully (71 pages)
