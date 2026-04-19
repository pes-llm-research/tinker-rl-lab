# Autoresearch: AI-Scientist-v2 Runnability on Tinker-RL-Lab

## Objective
Improve the success rate and quality of AI-Scientist-v2 experiments when targeting the tinker-rl-lab repository. The AI Scientist v2 uses best-first tree search (BFTS) to autonomously generate, execute, and refine research code. When run against this repo, experiments frequently fail because:
1. The `tinker` SDK (required by the repo's experiment template) is not importable in the execution environment
2. The agent's prompt doesn't advertise `tinker` as an available package
3. The experiment template (`tinker_grpo_rl.py`) may have robustness issues
4. Workspace setup doesn't properly expose repo code to the agent

## Metrics
- **Primary**: `runnability_score` (0-100, higher is better) — composite of environment readiness, template validity, and prompt completeness
- **Secondary**:
  - `deps_ready` (0/1) — all required packages importable
  - `template_parses` (0/1) — tinker_grpo_rl.py is syntactically valid
  - `template_runnable` (0/1) — a mocked version of the template can execute
  - `prompt_has_tinker` (0/1) — agent environment prompt mentions tinker

## How to Run
`./autoresearch.sh` — outputs `METRIC runnability_score=<0-100>` lines.

## Files in Scope
- `~/ai-scientist-v2/ai_scientist/treesearch/parallel_agent.py` — agent prompts, `_prompt_environment`
- `~/ai-scientist-v2/ai_scientist/treesearch/interpreter.py` — code execution subprocess
- `~/ai-scientist-v2/ai_scientist/treesearch/bfts_utils.py` — workspace setup
- `~/ai-scientist-v2/ai_scientist/treesearch/utils/config.py` — config loading, workspace prep
- `~/ai-scientist-v2/bfts_config.yaml` — BFTS hyperparameters
- `~/ai-scientist-v2/ai_scientist/ideas/tinker_grpo_rl.py` — experiment template
- `requirements.txt` — dependency declarations

## Off Limits
- Do NOT modify the core BFTS search algorithm (agent_manager.py, perform_experiments_bfts_with_agentmanager.py)
- Do NOT modify LLM backend code (backend_*.py, llm.py)
- Do NOT add new external dependencies to tinker-rl-lab unless strictly necessary
- Do NOT modify existing experiment results or paper content

## Constraints
- All changes must be backwards-compatible with existing AI-Scientist-v2 usage
- The benchmark must remain fast (< 10s per run)
- No hardcoded API keys or credentials

## What's Been Tried
- (Baseline) Initial runnability_score measured with missing tinker/orjson/datasets deps, prompt missing tinker reference
