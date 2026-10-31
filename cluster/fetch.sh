#!/usr/bin/env bash
# Laptop side: copy every run directory and log back from the node.
# Usage: cluster/fetch.sh <public-ip>
# Run it before releasing the lease - the node's disk is erased on release.
set -euo pipefail
ip=${1:?usage: cluster/fetch.sh <public-ip>}
root=$(cd "$(dirname "$0")/.." && pwd)
rsync -az --info=progress2 --exclude 20260829-005439-r1 \
      "ubuntu@$ip:mirage/runs/" "$root/runs/"
echo "fetched; now release the node in the console - the meter runs until you do"
