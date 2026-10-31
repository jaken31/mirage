#!/usr/bin/env bash
# Node side: decision 4's sweep, one run per GPU, each in its own tmux session.
# Usage: bash mirage/cluster/launch.sh [hours]   (default 2.25)
#
# Every run keeps decision 4a's stopping rule and the laptop's batch 16, so its
# numbers compare with r62 (no regularisation) and the laptop's regularised run.
# `timeout` is the lease's budget, not a stopping rule: a run it ends keeps its
# per-epoch model.pt and best.pt and can be `--resume`d on the laptop.
set -euo pipefail
cd ~/mirage
hours=${1:-2.25}

# gpu  config     dropout  - weight decay 0.1 throughout (the --train default)
runs=(
  "0 base     0.3"   # full size, stronger dropout
  "1 dyn256x6 0.1"
  "2 dyn256x6 0.3"
  "3 dyn192x4 0.1"
  "4 dyn192x4 0.3"
  "5 dyn128x4 0.1"   # ~21 tokens per parameter, Chinchilla-optimal
  "6 dyn128x4 0.0"   # control: does the optimal size need dropout at all
  "7 dyn128x2 0.1"
)
mkdir -p runs/logs
for r in "${runs[@]}"; do
  read -r gpu cfg p <<<"$r"
  log=runs/logs/gpu$gpu-$cfg-d$p.log
  tmux new-session -d -s "gpu$gpu" \
    "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=$gpu timeout ${hours}h \
     python -m mirage.dynamics --train --config mirage/configs/$cfg.json --dropout $p \
     > $log 2>&1"
  sleep 2   # run directories are named by the second
done
tmux ls
echo "logs: runs/logs/; watch with: tail -n 2 runs/logs/*.log"
