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

import re
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


REGISTER = ROOT / "docs" / "canonical_numbers.md"
NOTEBOOK = ROOT / "runs.jsonl"
# A register row: a backticked id alone in the first cell, then four more cells.
# The placeholder below is spelled `NUM-<id>` rather than with real-looking
# characters on purpose - CITE scans this file too, and a realistic example here
# would report itself as an undefined citation.
ROW = re.compile(r"^\|\s*`(NUM-[A-Z0-9-]+)`\s*\|(.*)\|\s*$")
CITE = re.compile(r"NUM-[A-Z0-9-]+")
# Definitions live above this heading; below it the same ids reappear as
# supersession chains, which are history rather than second definitions.
CHAINS = "## Superseded"


def check_register() -> list[str]:
    """Validate `docs/canonical_numbers.md` and every `NUM-` id that cites it.

    Four checks, all chosen because they cannot produce a false positive:

    1. no id is defined twice, and every row has all five cells
    2. every row carries a non-empty Source and Status - the provenance rule,
       enforced rather than trusted
    3. every `r<N>` source points at a `runs.jsonl` row that exists. This is the
       one that catches drift: rows are cited by index, so a citation to r45 when
       the notebook holds 41 is a dangling reference nobody would notice by eye
    4. every `NUM-` id cited anywhere in the tree is actually defined here, which
       is what makes renaming an entry safe

    ponytail: deliberately NOT checking that live docs are free of superseded
    values. It sounds like the obvious check and it is unimplementable without a
    per-site opt-out: measured 2026-08-29, the superseded values appear 60 times
    across the live docs and most are legitimate narration of the refutation that
    retired them. The chain table in the register is for humans, not for grep.
    """
    if not REGISTER.exists():
        return [f"{REGISTER.name} is missing"]

    problems: list[str] = []
    defined: dict[str, int] = {}
    sources: dict[str, str] = {}

    for n, line in enumerate(REGISTER.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith(CHAINS):
            break
        m = ROW.match(line)
        if m is None:
            continue
        fid, cells = m.group(1), [c.strip() for c in m.group(2).split("|")]
        if fid in defined:
            problems.append(f"{fid} defined twice, lines {defined[fid]} and {n}")
            continue
        if len(cells) != 4:
            problems.append(f"{fid} (line {n}) has {len(cells) + 1} cells, expected 5")
            continue
        defined[fid] = n
        _value, _what, source, status = cells
        if not source:
            problems.append(f"{fid} (line {n}) has no Source - every figure names where it came from")
        if not status:
            problems.append(f"{fid} (line {n}) has no Status")
        sources[fid] = source

    if not defined:
        return [f"{REGISTER.name} defines no NUM- ids - has the table format changed?"]

    rows = sum(1 for line in NOTEBOOK.read_text(encoding="utf-8").splitlines() if line.strip())
    for fid, source in sources.items():
        for ref in re.findall(r"\br(\d+)\b", source):
            if not 1 <= int(ref) <= rows:
                problems.append(
                    f"{fid} cites runs.jsonl row r{ref}, but the notebook has {rows} rows"
                )

    skip = {"build", "build-asan", ".git", "__pycache__", ".venv", "venv", "runs", "data"}
    cited: dict[str, set[str]] = {}
    for path in list(ROOT.rglob("*.md")) + list(ROOT.rglob("*.py")):
        if path == REGISTER or skip & set(path.parts):
            continue
        for fid in CITE.findall(path.read_text(encoding="utf-8", errors="ignore")):
            cited.setdefault(fid, set()).add(path.relative_to(ROOT).as_posix())

    for fid, where in sorted(cited.items()):
        if fid not in defined:
            problems.append(f"{fid} is cited by {', '.join(sorted(where))} but not defined")

    uncited = sorted(set(defined) - set(cited))
    print(f"  register: {len(defined)} ids, {len(cited)} cited, {len(uncited)} not yet cited")
    if uncited and len(uncited) < len(defined):
        print(f"  not yet cited: {', '.join(uncited[:8])}{' ...' if len(uncited) > 8 else ''}")
    return problems


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

    print(f"\n=== docs/canonical_numbers.md {'=' * 37}", flush=True)
    started = time.perf_counter()
    register = check_register()
    for problem in register:
        print(f"  {problem}")
    print(f"--- register: {'ok' if not register else f'{len(register)} problems'} "
          f"in {time.perf_counter() - started:.1f}s")

    print()
    if failed or register:
        parts = []
        if failed:
            parts.append(f"{', '.join(failed)} ({len(failed)} of {len(MODULES)})")
        if register:
            parts.append(f"{len(register)} register problems")
        print(f"FAILED: {'; '.join(parts)}")
        return 1
    print(f"all {len(MODULES)} self-checks ok, register clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
