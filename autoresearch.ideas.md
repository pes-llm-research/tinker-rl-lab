# Deferred Optimization Ideas

## AI Scientist v2 Runnability
- [x] Add tinker to agent prompt
- [x] Fix template main guard
- [x] Add graceful API key fallback
- [x] Add experiment_data.npy output
- [x] Add METRICS SUMMARY block
- [x] Increase BFTS timeout
- [x] Add pyproject.toml dependencies
- [x] Create local TRL template

## Paper Acceptance
- [x] Verify all reviewer weaknesses addressed
- [x] Verify critical scripts exist and run
- [x] Fix partial_correlation_zvf.py to read TSV
- [x] Produce real ZVF partial correlation results

## Future Ideas
1. **Improve local TRL template**: Use actual GRPOTrainer from TRL when available, rather than simplified policy gradient.
2. **Add more AI Scientist templates**: Create templates for process-reward models, tree-based sampling, code generation tasks.
3. **Run actual multi-seed experiments**: The survival_analysis.py shows only F3 survives strict filtering. Running real 5-seed experiments for F1, F2, F4, F5 would strengthen the paper.
4. **Integrate Tree-GRPO**: Add a template or experiment script for tree-based GRPO sampling (reviewer W15/Q7).
5. **Add ST-PPO comparison**: Implement token-level importance sampling diagnostic (reviewer W16).
6. **Auto-generate LaTeX tables**: Create a unified script that generates all paper tables from experiments/results/*.tsv.
7. **VLM plot review pipeline**: Ensure the AI Scientist's plotting code produces figures that pass VLM review.
