"""Run every module self-check in one command. Exit nonzero if any of them fails.

`python check.py`

Not a test framework, and deliberately not one - `docs/phase0_debt_checklist.md`
records "no test framework, no fixtures, self-checks per module" as a design
choice rather than an omission, and this does not reverse it. Each module still
owns its own `_self_check()` and is still runnable alone as
`python -m mirage.data`. The only thing missing was something that ran all five
without a human remembering the list, which is what this is.

ponytail: subprocesses rather than importing and calling `_self_check()`, so one
module's crash or `SystemExit` cannot take the runner down with it, and each gets
a clean interpreter. Serial, because `mirage.data` and `mirage.validator` each
sweep the 300,000-frame set and running them together would contend for the same
page cache and misreport both timings.
"""

import subprocess
import sys
import time
from pathlib import Path

# Cheap and data-free first, so a break surfaces in seconds instead of minutes.
# `config`, `logging` and `fsq` touch no dataset at all; `validator` and `data`
# sweep the full set, and fall back to the committed 40-frame fixture in
# `mirage/fixtures/` when `data/shards` is empty.
MODULES = ("config", "logging", "fsq", "validator", "data")

ROOT = Path(__file__).resolve().parent


def main() -> int:
    failed: list[str] = []
    for name in MODULES:
        print(f"\n=== python -m mirage.{name} {'=' * 44}", flush=True)
        started = time.perf_counter()
        # cwd=ROOT so this works when invoked from anywhere: `-m` resolves
        # `mirage` off the current directory, and a run from elsewhere would
        # otherwise die on ModuleNotFoundError rather than on a real failure.
        code = subprocess.run([sys.executable, "-m", f"mirage.{name}"], cwd=ROOT).returncode
        elapsed = time.perf_counter() - started
        print(f"--- mirage.{name}: {'ok' if code == 0 else f'FAILED rc={code}'} in {elapsed:.1f}s")
        if code != 0:
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)} ({len(failed)} of {len(MODULES)})")
        return 1
    print(f"all {len(MODULES)} self-checks ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
