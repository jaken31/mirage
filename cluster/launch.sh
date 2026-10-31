#!/usr/bin/env bash
# Node side: one run per GPU from a plan file, each in its own tmux session.
# Usage: bash mirage/cluster/launch.sh [plan] [hours]
#   plan:  cluster/plans/<name>.txt (default sweep), one run per line:
#          <gpu> <config> <extra --train flags...>; '#' starts a comment
#   hours: the budget every run is cut at (default 2.25)
#
# Every run keeps decision 4a's stopping rule and the laptop's batch 16, so its
# numbers compare with r62 and the runs before it. `timeout` is the lease's
# budget, not a stopping rule: a run it ends keeps its per-epoch model.pt and
# best.pt and can be `--resume`d on the laptop.
set -euo pipefail
cd ~/mirage
plan=${1:-cluster/plans/sweep.txt}
hours=${2:-2.25}

mkdir -p runs/logs
while read -r gpu cfg flags; do
  tag=$(echo "$flags" | tr -d '-' | tr ' .' '_-')
  log=runs/logs/gpu$gpu-$cfg-$tag.log
  tmux new-session -d -s "gpu$gpu" \
    "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=$gpu timeout ${hours}h \
     python -m mirage.dynamics --train --config mirage/configs/$cfg.json $flags > $log 2>&1"
  sleep 2   # run directories are named by the second
done < <(sed 's/#.*//' "$plan" | grep -v '^\s*$')
tmux ls
echo "logs: runs/logs/; watch with: tail -n 2 runs/logs/*.log"
