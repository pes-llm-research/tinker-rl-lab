#!/bin/bash
set -euo pipefail
# Apply tinker-rl-lab integration patches to ~/ai-scientist-v2

SRC="/Users/arvind/paper/tinker-rl-lab/ai-scientist-v2-integration"
DST="$HOME/ai-scientist-v2"

# Copy modified parallel_agent.py to add tinker to agent environment prompt
if [ -f "$SRC/ai_scientist/treesearch/parallel_agent.py" ]; then
    cp "$SRC/ai_scientist/treesearch/parallel_agent.py" \
       "$DST/ai_scientist/treesearch/parallel_agent.py"
    echo "PATCHED parallel_agent.py"
fi

# Future patches go here...
echo "Patches applied."
