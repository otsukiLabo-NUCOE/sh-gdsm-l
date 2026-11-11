#!/usr/bin/env bash
# Resume the Claude API experiments (E-L) after a reboot.
# Finished (arm, seed) cells / ToolMaze runs are kept and skipped.
cd "$(dirname "$0")"
mkdir -p EL
for g in "1,2" "3,4" "5,6" "7,8" "9,10"; do
  n=${g/,/_}
  python -u llm_experiment.py --backend anthropic --model claude-sonnet-5-5 --arms A,B,D \
    --seed-list $g --episodes 30 --out EL/exp4_sonnet_s$n.json >> EL/exp4_sonnet_s$n.log 2>&1 &
done
for m in P2 P4; do
  python -u toolmaze_pilot.py --backend anthropic --model claude-sonnet-5-5 --modes $m \
    --tasks-per-cell 50 --arms llm,hybrid --out EL/toolmaze_${m}_sonnet.json >> EL/toolmaze_${m}_sonnet.log 2>&1 &
done
wait
echo "E-L (Sonnet) finished"
