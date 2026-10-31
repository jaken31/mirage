#!/usr/bin/env bash
# Node side: as each GPU's run ends, start the next queued run on it, so a
# lease billed for every GPU never idles one. Usage, after launch.sh:
#   tmux new-session -d -s filler 'bash mirage/cluster/fill.sh <deadline HH:MM, node clock>'
#
# The queue is second seeds of the sweep's settings, best held-out loss first,
# so the spread between seeds is measured where a decision would rest on it.
# Every run ends by the deadline; one with under MIN_MIN minutes left is not started.
set -euo pipefail
cd ~/mirage
deadline=$(date -d "${1:?usage: fill.sh <deadline HH:MM>}" +%s)
MIN_MIN=20

# config    dropout seed
queue=(
  "dyn192x4 0.1 1"
  "dyn256x6 0.1 1"
  "dyn128x4 0.1 1"
  "dyn256x6 0.3 1"
  "dyn128x2 0.1 1"
  "dyn192x4 0.3 1"
  "dyn128x4 0.0 1"
  "base     0.3 1"
)
while ((${#queue[@]})); do
  for gpu in 0 1 2 3 4 5 6 7; do
    ((${#queue[@]})) || break
    tmux has-session -t "=gpu$gpu" 2>/dev/null && continue
    left=$(( (deadline - $(date +%s)) / 60 ))
    if (( left < MIN_MIN )); then
      echo "$(date +%T) $left min left, not starting: ${queue[*]}"
      exit 0
    fi
    read -r cfg p seed <<<"${queue[0]}"
    queue=("${queue[@]:1}")
    log=runs/logs/gpu$gpu-$cfg-d$p-s$seed.log
    tmux new-session -d -s "gpu$gpu" \
      "source .venv/bin/activate && CUDA_VISIBLE_DEVICES=$gpu timeout ${left}m \
       python -m mirage.dynamics --train --config mirage/configs/$cfg.json --dropout $p \
       --seed $seed > $log 2>&1"
    echo "$(date +%T) gpu$gpu: $cfg dropout $p seed $seed, $left min budget"
    sleep 2   # run directories are named by the second
  done
  sleep 30
done
