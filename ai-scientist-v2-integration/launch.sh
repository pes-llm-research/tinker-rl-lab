#!/bin/bash
set -euo pipefail
# Convenience launcher for AI-Scientist-v2 on tinker-rl-lab

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Apply patches
bash "$SCRIPT_DIR/patch.sh"

cd "$HOME/ai-scientist-v2"

# Default: use the Tinker template
IDEA="${1:-ai_scientist/ideas/tinker_grpo_rl.json}"
MODEL_WRITEUP="${MODEL_WRITEUP:-o1-preview-2024-09-12}"
MODEL_CITATION="${MODEL_CITATION:-gpt-4o-2024-11-20}"

echo "Launching AI Scientist v2 with idea: $IDEA"
echo "Writeup model: $MODEL_WRITEUP"
echo "Citation model: $MODEL_CITATION"

python launch_scientist_bfts.py \
  --load_ideas "$IDEA" \
  --load_code \
  --model_writeup "$MODEL_WRITEUP" \
  --model_citation "$MODEL_CITATION" \
  --num_cite_rounds 20
