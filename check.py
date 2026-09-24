"""Run every module self-check in one command. Exits nonzero if any fails.

`python check.py`

Deliberately not a test framework: "no test framework, per-module self-checks"
is a recorded design choice (`docs/phase0_debt_checklist.md`), and this does
not reverse it. Each module keeps its own `_self_check()` and still runs alone,
e.g. `python -m mirage.data`. This just runs all six so nobody has to remember
the list. It also checks the numbers register (`docs/canonical_numbers.md`).

ponytail: runs each check as a subprocess instead of importing it, so one
module's crash or `SystemExit` cannot kill the runner, and each gets a clean
interpreter. One at a time, because `mirage.data` and `mirage.validator` each
read the whole 300,000-frame dataset, and running them together would compete
for the same file cache and distort both timings.
"""

import re
import subprocess
import sys
import time
from pathlib import Path

# Fast, data-free checks first, so a failure shows up in seconds, not minutes.
# `config`, `logging` and `fsq` touch no dataset; `validator`, `data` and
# `dynamics` read the full dataset, or the committed 40-frame fixture in
# `mirage/fixtures/` when `data/shards` is empty. `dynamics` also reads R1's
# token cache when there is one.
MODULES = ("config", "logging", "fsq", "validator", "data", "dynamics")

ROOT = Path(__file__).resolve().parent


REGISTER = ROOT / "docs" / "canonical_numbers.md"
NOTEBOOK = ROOT / "runs.jsonl"
# A register row: a backticked id alone in the first cell, then four more cells.
# The placeholder is written `NUM-<id>`, not a realistic id, on purpose: CITE
# scans this file too, and a realistic example would be reported as an
# undefined citation.
ROW = re.compile(r"^\|\s*`(NUM-[A-Z0-9-]+)`\s*\|(.*)\|\s*$")
CITE = re.compile(r"NUM-[A-Z0-9-]+")
# Definitions are above this heading. Below it the same ids reappear in the
# history of replaced values, which are not second definitions.
CHAINS = "## Superseded"


def check_register() -> list[str]:
    """Validate `docs/canonical_numbers.md` and every `NUM-` id that cites it.

    Four checks, each chosen because it cannot raise a false alarm:

    1. no id is defined twice, and every row has all five cells
    2. every row has a non-empty Source and Status, so "every number names where
       it came from" is enforced rather than trusted
    3. every `r<N>` source points at a `runs.jsonl` row that exists. This catches
       drift: rows are cited by position, so a citation to row 45 when the file
       has 41 rows is a broken reference nobody would spot by eye
    4. every `NUM-` id cited anywhere in the repo is defined here, which is what
       makes renaming an entry safe

    Check 4 also fires on a bare *group prefix*, a group name with a trailing
    hyphen and nothing after it, meant as "every row in this group". That is on
    purpose and was kept after it fired for real: writing out each id is clearer
    for readers and keeps every id searchable. For the same reason, the prefix
    form never appears literally in this file, since CITE scans it too.

    ponytail: deliberately does NOT check that live docs are free of replaced
    values. It sounds obvious but cannot work without a per-line opt-out: the
    replaced values appear about 60 times across the live docs, and most are
    legitimate explanations of why they were replaced. The history table in the
    register is for people, not for grep.
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
        # cwd=ROOT so this works from any directory: `-m` finds `mirage` in
        # the current directory, so running from elsewhere would otherwise fail
        # with ModuleNotFoundError instead of testing anything.
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
