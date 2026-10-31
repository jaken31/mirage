#!/usr/bin/env bash
# Node side: the repo's pinned stack (requirements.txt) on a fresh Ubuntu node,
# then proof it works before any money goes on training. Run by push.sh.
set -euo pipefail
cd ~/mirage

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
command -v tmux >/dev/null || sudo apt-get install -y tmux

# Python 3.14 and torch 2.9.1+cu130, the stack every runs.jsonl number was
# measured on; uv fetches the interpreter, so the image's own Python is unused.
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install -r requirements.txt

# The GPU is reachable and this build has kernels for it: a CPU-only torch or
# a wheel without this architecture fails here, not an hour into a run.
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch sees no GPU"
cap = torch.cuda.get_device_capability()
print(torch.__version__, torch.cuda.get_device_name(), f"sm_{cap[0]}{cap[1]}",
      "arch list", torch.cuda.get_arch_list(), "gpus", torch.cuda.device_count())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); (x @ x).sum().item()
print("bf16 matmul ok")
EOF

# The data arrived intact: the self-check verifies the shards' and the token
# cache's sha256 against their manifests.
python -m mirage.dynamics | tail -1
