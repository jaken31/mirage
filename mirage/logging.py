"""One `log(dict)` that always writes a jsonl file, and copies to W&B if asked.

There are three kinds of record in this project. `runs.jsonl` at the repo root
is the hand-written lab notebook, one line per *decision*; no tool writes it
and this file must not touch it. This module writes the per-run metrics. W&B,
when enabled, is only a viewer over those metrics.

The jsonl file is the source of truth, so the history does not depend on any
account, and benchmark tables can be generated from it by script. W&B is an
option, never a dependency: this module imports and runs without `wandb`
installed. The self-check tests the W&B copy offline against wandb 0.29.0.

Every record carries the run id and any hashes the caller passes, so any single
line from the middle of a run says which run and which config produced it,
without anyone having to remember to write that down.

**Do not call this inside a timed region.** Frame-time targets are judged on
the slowest 1% of frames, and an occasional 1 ms write lands right there. Timed
benchmark loops should record into a preallocated array and write afterwards.

    with Run("r0", {"tokenizer_hash": cfg.tokenizer_hash}) as run:
        run.log({"step": step, "loss": float(loss)})
"""

import os
import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent

# Next to runs.jsonl, deliberately not inside it. The trailing slash in
# .gitignore stops `runs/` from also ignoring runs.jsonl, but the two similar
# names are a real trap; read the note there before "fixing" either.
RUNS_DIR = ROOT / "runs"

# How long the W&B login prompt may block a run before it counts as a failure.
# Only reached on a terminal with no key set; every other credential problem
# fails in under a second. Long enough to paste a key, short enough that an
# unattended run gives up instead of waiting forever.
WANDB_LOGIN_TIMEOUT_S = 30


def _jsonable(value: Any) -> Any:
    """Fallback conversion for values `json` cannot serialise.

    Handles what actually shows up in a training loop: numpy scalars and arrays,
    torch tensors, and `Path`. Anything else becomes its `repr` instead of
    raising, because losing a whole run's log over one odd field is worse than
    one ugly string.
    """
    for attr in ("item", "tolist"):  # numpy scalar / 0-d array, torch tensor
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(value, Path):
        # as_posix, not str: this runs on Windows, and str() would put
        # backslashes in a log meant to be read anywhere.
        return value.as_posix()
    return repr(value)


def git_sha() -> str | None:
    """The working tree's commit, or None outside a repo. Never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


class Run:
    """One run's jsonl metrics file, plus a `meta.json` describing the run.

    The directory is named by run id, not config hash: two runs with the same
    config but different seeds share a hash yet give different results, so
    hash-named directories would overwrite each other. The hash goes *inside*
    each record instead.

    Creating a run whose directory already exists raises instead of appending.
    Two processes mixing lines into one metrics file would be a silent bug.
    """

    def __init__(
        self,
        name: str,
        hashes: Mapping[str, str] | None = None,
        config: Mapping[str, Any] | None = None,
        root: Path | str | None = None,
        wandb_project: str | None = None,
    ) -> None:
        self.started = time.time()
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.started))}-{name}"
        self.hashes = dict(hashes or {})

        self.dir = Path(root or RUNS_DIR) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)
        self.path = self.dir / "metrics.jsonl"
        self._file = None

        try:
            self._file = self.path.open("w", encoding="utf-8", newline="\n")
            (self.dir / "meta.json").write_text(
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                        "git_sha": git_sha(),
                        "hashes": self.hashes,
                        "config": config,
                    },
                    indent=1, default=_jsonable,
                ) + "\n",
                encoding="utf-8", newline="\n",
            )

            # Imported here, not at the top of the module: a top-level `import wandb`
            # would make an optional viewer a hard dependency of every run.
            #
            # Tested against wandb 0.29.0: the offline init/log/finish path (by
            # `_self_check`), the credential failures (all fail here, at init, before
            # step 0; see the `wandb.login` note below), and a real upload (by
            # `--network`, which reads the history back through `wandb.Api()`).
            # Resume, artifacts, media and a network drop mid-run are deliberately
            # not tested; the verification log says why.
            self._wandb = None
            if wandb_project is not None:
                import wandb  # noqa: PLC0415

                # `wandb.init` logs in implicitly, and with no key set that means an
                # **interactive prompt with no timeout**: a run started from a
                # terminal waits at "Enter your choice:" forever, having logged
                # nothing. That is worse than crashing, and it is the default.
                # `Settings(login_timeout=)` does not affect that prompt; only an
                # explicit `login(timeout=)` does (tested on 0.29.0).
                #
                # `login` returning False is the other quiet failure: `init` would
                # then succeed and send nothing. So raise instead. A W&B copy that is
                # silently off is worse than a run that refuses to start, because the
                # run it was meant to record is the multi-hour one. False comes with
                # no reason (the prompt timed out, the user declined to paste a key,
                # or a W&B run is already active in this process), so the message
                # lists all three rather than guess. Telling them apart would mean
                # reading wandb's private internals.
                #
                # Offline and disabled modes are exempt on purpose: neither talks to
                # the server, so neither needs a key, and the offline self-check
                # relies on that. We ask wandb which mode is active instead of
                # listing mode names here. In 0.29.0 `_offline` means "offline" or
                # "dryrun" and `_noop` means "disabled" (checked for all six mode
                # names; the command is in the verification log of
                # `docs/world_model_architecture.md`), so the "dryrun" alias is
                # covered too. These attributes are
                # private and an upgrade could rename them, which is why
                # `_bad_credentials_check` fails on an `AttributeError`. Accepted
                # side effect: wandb reports "server unreachable" as an auth error,
                # so a brief outage also stops the run here, labelled as auth. The
                # W&B copy is optional, so re-run without `wandb_project`.
                wandb_settings = wandb.setup().settings
                if not (
                    wandb_settings._offline or wandb_settings._noop
                ) and not wandb.login(timeout=WANDB_LOGIN_TIMEOUT_S):
                    # A failed `login` does more than return False: it changes
                    # the mode of wandb's process-wide session (`disabled` on
                    # timeout, `offline` when declined; see 0.29.0
                    # `sdk/wandb_login.py`). The guard above reads exactly that,
                    # so a second `Run(..., wandb_project=...)` in this process
                    # (a notebook cell, a retry, a sweep) would be exempted and
                    # silently send nothing. Tearing the session down makes the
                    # next `wandb.setup()` re-read the mode from the
                    # environment, so a retry fails the same way. A teardown
                    # failure is printed, not raised, so it cannot hide the
                    # login error the user needs to read.
                    #
                    # Skip the teardown when a run is already active: `login`
                    # returns False in that case without touching the session,
                    # and `teardown` would finish that caller's live run with
                    # exit code 0 while it is still training.
                    if wandb.run is None:
                        try:
                            wandb.teardown()
                        except Exception as teardown_exc:  # noqa: BLE001 - must not mask the login error
                            print(f"warning: wandb.teardown() failed: {teardown_exc}",
                                  file=sys.stderr)
                    raise RuntimeError(
                        "W&B is online but the login did not complete, so the "
                        "mirror would be off. Either no API key is configured and "
                        f"the prompt went unanswered within {WANDB_LOGIN_TIMEOUT_S}s "
                        "or was declined - set WANDB_API_KEY, or run `wandb login`, "
                        "or set WANDB_MODE=offline, never a key in a config or the "
                        "repo - or a W&B run is already active in this process, "
                        "which is not a credential problem: finish that run first."
                    )

                # x_disable_stats turns off wandb's background CPU/GPU/disk
                # sampler, which would quietly slow the worst frame times during
                # latency measurements.
                self._wandb = wandb.init(
                    project=wandb_project, name=self.run_id,
                    config=dict(config or {}) | self.hashes,
                    settings=wandb.Settings(x_disable_stats=True),
                )
        except BaseException:
            # If `__init__` raises, `__exit__` never runs, so the open file and
            # the new directory would be left behind. A retry within the same
            # second would then build the same timestamped `run_id` and fail on
            # `mkdir(exist_ok=False)`, which looks like the two-writers bug from
            # the class docstring rather than a login failure. So clean up here.
            #
            # Removing the directory is safe because `exist_ok=False` proved
            # this call created it, so nothing else can be in it. The two files
            # are removed by name and the directory with `rmdir`, not a
            # recursive delete, so an unexpected file stops the cleanup instead
            # of being swept away. A cleanup failure is printed, not raised, so
            # the login error stays visible. `close()` is inside the same guard
            # for the same reason; `self._file` is `None` if opening it failed.
            try:
                if self._file is not None:
                    self._file.close()
                (self.dir / "meta.json").unlink(missing_ok=True)
                self.path.unlink(missing_ok=True)
                self.dir.rmdir()
            except OSError as cleanup_exc:
                print(f"warning: could not remove {self.dir}: {cleanup_exc}",
                      file=sys.stderr)
            raise

    def log(self, record: Mapping[str, Any]) -> dict:
        """Append one record. Returns the line as written, parsed back.

        Returns the parsed line, not the input dict, on purpose. Callers pass
        numpy values that `_jsonable` converts while writing, so the input and the
        line on disk differ, and checking against the input would check something
        that was never written.

        Flushed after every line: a run that dies at epoch 12 should still have
        epochs 1-11 on disk, which is worth the extra writes. `t` is seconds since
        the run started, so a record reads fine without knowing the start time.
        """
        line = json.dumps(
            {
                "t": round(time.time() - self.started, 3),
                "run_id": self.run_id,
                **self.hashes,
                **record,
            },
            default=_jsonable,
        )
        self._file.write(line + "\n")
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(dict(record))
        return json.loads(line)

    # wandb treats a *missing* `exit_code` as 0, so a crashed run used to show as
    # `finished` on the server. `__exit__` now passes the real code. The default
    # stays 0 because `close()` is also called directly. `_self_check` tests what
    # this passes with a stub (no wandb, no account); only the offline branch calls
    # the real `finish(exit_code=0)`.
    def close(self, exit_code: int = 0) -> None:
        if not self._file.closed:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish(exit_code=exit_code)
            self._wandb = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Returns None on purpose: a crashed run must be marked failed *and*
        # still raise. Swallowing the exception would turn every training crash
        # into a silent early return.
        self.close(exit_code=1 if exc_type is not None else 0)


def _self_check() -> None:
    """Checks this module works, in a throwaway directory.

    Four parts, in order: the jsonl path, which must work without W&B at all;
    exit codes, using a stub so no wandb package or account is needed; the W&B
    copy in offline mode against a local directory; and bad credentials
    (`_bad_credentials_check`), which contacts the server. The W&B parts are
    skipped only if wandb is not installed.

    Real uploads are not tested here because they need a key; run
    `--network <project>` for that. The verification log at the end of
    `docs/world_model_architecture.md` records what each part measured.
    """
    import tempfile

    import numpy as np
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        hashes = {"tokenizer_hash": "978246d7157caa27", "data_hash": "18a76531aaa8b609"}
        with Run("selfcheck", hashes, config={"lr": 3e-4}, root=tmp) as run:
            written = [run.log({"step": s, "loss": 1.0 / (s + 1)}) for s in range(5)]
            path, run_dir, run_id = run.path, run.dir, run.run_id

        assert written[-1]["run_id"] == run_id and written[-1]["step"] == 4
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 5, f"{len(lines)} lines for 5 log calls"
        rows = [json.loads(line) for line in lines]
        for r in rows:
            assert r["run_id"] == run_id, "a record does not name its run"
            for k, v in hashes.items():
                assert r[k] == v, f"a record does not carry {k}"
        assert [r["step"] for r in rows] == list(range(5)), "records are out of order"
        assert rows == written, "log() returned something other than what it wrote"
        assert all(rows[i]["t"] <= rows[i + 1]["t"] for i in range(4)), "t is not monotone"
        print(f"jsonl: {len(rows)} records, each carrying run_id and "
              f"{len(hashes)} hashes, in order")

        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["run_id"] == run_id and meta["hashes"] == hashes
        print(f"meta.json: run_id, git_sha {str(meta['git_sha'])[:8]}, hashes, config")

        # A run must not reuse an existing run directory, or two processes
        # would silently mix lines into one file.
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            raise AssertionError("re-creating an existing run directory did not raise")
        except FileExistsError:
            print("a second run cannot reopen an existing run directory")

        # The metrics file must be readable by pandas, which is what will read it.
        df = pd.read_json(path, lines=True)
        assert len(df) == 5 and df["step"].tolist() == list(range(5))
        assert set(hashes) <= set(df.columns), "pandas lost the hash columns"
        print(f"pandas.read_json(lines=True): {df.shape[0]} rows x {df.shape[1]} columns, "
              f"columns {sorted(df.columns)}")

        # Conversion of the value types a training loop actually produces.
        with Run("coerce", root=tmp) as run2:
            got = run2.log({"a": np.float32(0.5), "b": np.arange(3), "c": Path("x/y"),
                            "d": object()})
        assert got["a"] == 0.5, "a numpy scalar did not survive"
        assert got["b"] == [0, 1, 2], "a numpy array did not survive"
        assert got["c"] == "x/y", "a Path did not survive"
        assert isinstance(got["d"], str), "an unserialisable field was not coerced"
        on_disk = json.loads(run2.path.read_text(encoding="utf-8").strip())
        assert on_disk == got, "the coerced return value differs from the line on disk"
        print("coercion: numpy scalar, numpy array, Path and an opaque object all survive")

        # A crashed run must call `finish()` with a nonzero exit_code, a clean
        # one with 0, and the exception must still propagate. A stub stands in
        # for wandb, so this needs no network or credentials.
        class _StubWandb:
            def __init__(self) -> None:
                self.exit_codes: list[int] = []

            def finish(self, exit_code: int = 0) -> None:
                self.exit_codes.append(exit_code)

        run3 = Run("exitcode-crash", root=tmp)
        stub = run3._wandb = _StubWandb()
        try:
            with run3:
                raise ValueError("boom")
        except ValueError:
            pass
        else:
            raise AssertionError("__exit__ swallowed the exception")
        assert stub.exit_codes == [1], f"crash did not reach finish() with exit_code=1: {stub.exit_codes}"

        run4 = Run("exitcode-clean", root=tmp)
        stub2 = run4._wandb = _StubWandb()
        with run4:
            pass
        assert stub2.exit_codes == [0], f"clean exit did not reach finish() with exit_code=0: {stub2.exit_codes}"

        run5 = Run("exitcode-direct-close", root=tmp)
        stub3 = run5._wandb = _StubWandb()
        run5.close()
        assert stub3.exit_codes == [0], f"close() with no arguments changed behaviour: {stub3.exit_codes}"
        print("exit_code: crash -> finish(exit_code=1) with the exception still propagating, "
              "clean exit and a bare close() -> finish(exit_code=0)")

    # The W&B copy, end to end in **offline** mode. That still tests the real
    # risk: that `wandb.init(...)` or `Settings(x_disable_stats=...)` does not
    # match the installed version and crashes at the start of a multi-hour run.
    # Offline makes the same calls against a local directory. It does NOT test
    # upload or what the server sees.
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        print("W&B mirror NOT exercised - wandb is absent from this environment")
    else:
        prev = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "offline"
        try:
            # ignore_cleanup_errors is needed: on Windows, wandb 0.29.0 keeps
            # `wandb/offline-run-*/logs/debug-internal.log` open after
            # `finish()`, so deleting the directory raises WinError 32 and
            # would fail a check that already passed.
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
                os.environ["WANDB_DIR"] = tmp
                with Run("wandb", root=Path(tmp), hashes={"h": "0" * 8},
                         config={"epochs": 1}, wandb_project="mirage-selfcheck") as run3:
                    assert run3._wandb is not None, "wandb_project was passed but no run was made"
                    for i in range(3):
                        run3.log({"step": i, "loss": 1.0 / (i + 1)})
                assert run3._wandb is None, "finish() did not clear the wandb handle"
                lines = run3.path.read_text(encoding="utf-8").strip().splitlines()
                assert len(lines) == 3, "the jsonl path lost lines while mirroring"
            print(f"W&B mirror ok, offline, wandb {wandb.__version__}: init with "
                  f"Settings(x_disable_stats=True), 3 records, finish - and the "
                  f"jsonl path is unaffected. Upload is NOT checked")
        finally:
            os.environ.pop("WANDB_DIR", None)
            if prev is None:
                os.environ.pop("WANDB_MODE", None)
            else:
                os.environ["WANDB_MODE"] = prev

        # Not just tidiness: `wandb.setup()` caches a process-wide session on
        # first use and then **ignores** later WANDB_MODE changes. Without this
        # teardown the next check inherits `offline` from above, skips the online
        # guard it exists to test, and passes for the wrong reason. It did once.
        wandb.teardown()
        _bad_credentials_check()

    print("logging self-check ok")


def _bad_credentials_check() -> None:
    """A wrong key must make `Run` fail at construction, before any record.

    The danger is not a crash. It is the *quiet* failure, where the W&B copy
    switches itself off and the run trains for hours reporting to nothing.
    Wrong-key and no-key both fail at `Run` construction. No-key on a terminal
    is the case that used to hang, but it needs a terminal to test, so this
    tests the wrong-key case.

    **What this cannot tell you:** whether the key was the cause. wandb raises
    `AuthenticationError` both when the server rejects a key and when it cannot
    be reached (on purpose; see `_verify_login`). So with no network this sees
    the same error as a rejection. Catching only `AuthenticationError` /
    `UsageError` was tried and dropped for that reason. Do not retry it, and do
    not match on the message text either: a check that breaks when wandb
    rewords a string is worse than an honest broad one.

    So it proves `Run` refuses to start before step 0 instead of running with
    the W&B copy silently off. It does not prove the key caused the refusal.
    """
    import tempfile

    import wandb  # noqa: PLC0415

    prev_key, prev_mode = os.environ.get("WANDB_API_KEY"), os.environ.get("WANDB_MODE")
    os.environ["WANDB_API_KEY"] = "0" * 40  # syntactically valid, no such account
    os.environ.pop("WANDB_MODE", None)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            os.environ["WANDB_DIR"] = tmp
            try:
                Run("badkey", root=Path(tmp), wandb_project="mirage-selfcheck")
            except Exception as exc:  # noqa: BLE001 - see the docstring
                # Only the constructor could have created a `*-badkey`
                # directory here, so any that remains was left behind by its
                # failure path. That leak would make a retry in the same second
                # fail on `mkdir(exist_ok=False)`.
                orphans = [str(d) for d in Path(tmp).glob("*-badkey")]
                assert not orphans, f"failed Run() left its directory behind: {orphans}"
                # A plain programming error is not a credential refusal. The
                # online guard reads two private wandb attributes, and this is
                # the only check that exercises them. Without this line, a
                # rename in wandb would show up as a passing "refused to
                # construct" for a W&B copy that cannot start at all.
                assert not isinstance(exc, (AttributeError, TypeError, NameError)), (
                    f"Run() failed with {type(exc).__name__}, which is mirage's "
                    f"own bug, not a credential refusal: {exc}"
                )
                print(f"bad credentials: Run() refused to construct, before "
                      f"step 0, with {type(exc).__name__} - a type consistent "
                      f"with either a rejected key or an unreachable server, "
                      f"leaving no run directory behind")
            else:
                raise AssertionError(
                    "a 40-zero API key was accepted - the mirror is silently off"
                )
    finally:
        os.environ.pop("WANDB_DIR", None)
        for key, prev in (("WANDB_API_KEY", prev_key), ("WANDB_MODE", prev_mode)):
            if prev is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = prev
        # Same reason as the teardown before this check: this one leaves a
        # cached session holding the fake key, which the next check would
        # silently inherit. It runs last, in a `finally`, so a failure here is
        # printed rather than raised and cannot hide this check's own result.
        try:
            wandb.teardown()
        except Exception as teardown_exc:  # noqa: BLE001 - must not mask the check
            print(f"warning: wandb.teardown() failed: {teardown_exc}",
                  file=sys.stderr)


def _network_check(project: str) -> None:
    """What offline cannot check: a real run, uploaded and read back.

    Deliberately tiny: three records, no GPU. Reading the history back through
    the public API is the point, because `finish()` returning only shows that
    nothing raised, not that anything arrived.
    """
    import tempfile

    import wandb

    assert os.environ.get("WANDB_MODE") in (None, "online"), "WANDB_MODE is not online"
    records = [{"step": i, "loss": 1.0 / (i + 1)} for i in range(3)]
    hashes = {"h": "0" * 8}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["WANDB_DIR"] = tmp
        try:
            with Run("netcheck", hashes, config={"epochs": 1},
                     root=Path(tmp), wandb_project=project) as run:
                assert run._wandb is not None, "wandb_project was passed but no run was made"
                path, run_id = run.path, run.run_id
                url, wid, entity = run._wandb.url, run._wandb.id, run._wandb.entity
                print(f"authenticated as {entity!r}, run {url}")
                for r in records:
                    run.log(r)
        finally:
            os.environ.pop("WANDB_DIR", None)

        # The local log is the source of truth and the W&B copy must not change it.
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == len(records), f"{len(rows)} local lines for {len(records)} logs"
        for row, rec in zip(rows, records):
            assert {k: row[k] for k in rec} == rec, "a local record does not match what was logged"
            assert row["run_id"] == run_id, "a local record does not name its run"
            for k, v in hashes.items():
                assert row[k] == v, f"a local record does not carry {k}"
        print(f"local jsonl intact underneath the mirror: {len(rows)} records, "
              f"each carrying run_id and {', '.join(hashes)}")

    # The server's view, through the public API, as another process would see it.
    api_run = wandb.Api().run(f"{entity}/{project}/{wid}")
    history = list(api_run.scan_history(keys=["step", "loss"]))
    assert api_run.state == "finished", f"server reports state {api_run.state!r}"
    assert len(history) == len(records), f"server has {len(history)} rows for {len(records)} logs"
    for got, rec in zip(history, records):
        assert got["step"] == rec["step"] and got["loss"] == rec["loss"], "server row differs"
    print(f"server-side: state {api_run.state!r}, {len(history)} rows read back "
          f"through wandb.Api(), values match")
    print(f"W&B networked check ok, wandb {wandb.__version__}: {url}")


if __name__ == "__main__":
    # `--network <project>` needs a real key, from the environment or `wandb
    # login`, never from a file in this repo. Everything else runs without one.
    # Any other argument is rejected: falling through to the offline check would
    # answer a typo like `--netowrk` with a pass from a check that never
    # contacted the server.
    usage = "usage: python -m mirage.logging [--network <project>]"
    if sys.argv[1:2] == ["--network"]:
        if len(sys.argv) != 3 or not sys.argv[2].strip():
            raise SystemExit(usage)
        _network_check(sys.argv[2])
    elif sys.argv[1:]:
        raise SystemExit(usage)
    else:
        _self_check()
