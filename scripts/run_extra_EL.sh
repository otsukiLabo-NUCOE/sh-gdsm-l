#!/usr/bin/env bash
# Extra experiments (1) ToolMaze P2 with Opus 5.5 (held-out model for hybrid-h)
#                   (2) Exp6 S4 with Sonnet 5.5, seeds 4-12
# Run ONLY on the author's instruction. Uses the Anthropic API (paid).
cd "$(dirname "$0")"
mkdir -p EL
for a in llm hybrid-h; do
  python -u toolmaze_pilot.py --backend anthropic --model claude-opus-5-5 --modes P2 \
    --tasks-per-cell 100 --arms $a --out EL/toolmaze_P2_opus_$a.json >> EL/toolmaze_P2_opus_$a.log 2>&1 &
done
for st in 4 7 10; do
  python -u recovery_experiment.py --llm --backend anthropic --model claude-sonnet-5-5 \
    --scenarios S4 --llm-arms C,A,B,C+H --llm-seed-start $st --llm-seeds 3 --episodes 12 \
    --out EL/exp6_sonnet_S4_s$st.json >> EL/exp6_sonnet_S4_s$st.log 2>&1 &
done
wait
echo "extra EL finished"
