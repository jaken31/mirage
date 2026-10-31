#!/usr/bin/env bash
# Laptop side: copy the code, the dataset shards and R1's tokenizer run to a
# leased node, then set it up there. Usage: cluster/push.sh <public-ip>
#
# The node's disk is erased when the lease ends, so this runs once per lease.
# The meter runs from the lease, so everything here is timed.
set -euo pipefail
ip=${1:?usage: cluster/push.sh <public-ip>}
host=ubuntu@$ip
root=$(cd "$(dirname "$0")/.." && pwd)
t0=$SECONDS
# A fresh node per lease means a new host key every time; accept it once.
export RSYNC_RSH="ssh -o StrictHostKeyChecking=accept-new"

# rsync creates only the last directory of a destination, and the code copy
# below excludes data/ and runs/, so make both targets first.
ssh -o StrictHostKeyChecking=accept-new "$host" 'mkdir -p mirage/data/shards mirage/runs/20260829-005439-r1'
rsync -az --exclude .venv --exclude runs --exclude data --exclude build \
      --exclude .claude-flow --exclude .git "$root/" "$host:mirage/"
rsync -az --info=progress2 "$root/data/shards/" "$host:mirage/data/shards/"
rsync -az "$root/runs/20260829-005439-r1/" "$host:mirage/runs/20260829-005439-r1/"
echo "copied in $((SECONDS - t0)) s"

ssh -o StrictHostKeyChecking=accept-new "$host" 'bash mirage/cluster/node_setup.sh'
echo "pushed and set up in $((SECONDS - t0)) s; next: ssh $host 'bash mirage/cluster/launch.sh'"
