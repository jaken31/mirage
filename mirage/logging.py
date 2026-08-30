"""One `log(dict)` that always writes jsonl, and mirrors to W&B when asked.

Three layers of observability are named in the architecture doc; this file is
the second. The first is `runs.jsonl` at the repo root - the hand-authored E-5
lab notebook, one line per *decision*, which no tool produces and which this
file must not touch. The third is W&B, which is a viewer over this one.

jsonl stays the source of truth, so the history survives independently of any
account, and F-17's ladder table is a jsonl-to-markdown script rather than a
screenshot. W&B is a flag, never a dependency: the whole module imports and runs
with `wandb` absent from the environment; the mirror itself is verified
offline against 0.29.0 by the self-check.

Every record carries the run id and whatever hashes the caller names, so E-4
("reproducible from a config hash") and E-5 hold by construction rather than by
remembering to write them down. A single line out of the middle of a training
run identifies which run and which config produced it.

**Do not call this inside a timed region.** P-2 and P-4 are tail-latency
requirements and an occasional 1 ms write lands in p99, not p50; the Phase 4
bench loop records into a preallocated array and writes afterwards.

    with Run("r0", {"tokenizer_hash": cfg.tokenizer_hash}) as run:
        run.log({"step": step, "loss": float(loss)})
"""

import os
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent.parent

# Sibling of runs.jsonl, and deliberately not inside it. The trailing slash in
# .gitignore keeps `runs/` from matching the file, but the two names sitting
# side by side is a real trap - read the note there before "fixing" either.
RUNS_DIR = ROOT / "runs"

# How long the W&B login prompt may block a training run before it is treated as
# a failure. Only reached on a terminal with no key configured; every other
# credential outcome fails in under a second. Generous enough to paste a key
# into, short enough that an unattended run gives up rather than sitting there.
WANDB_LOGIN_TIMEOUT_S = 30


def _jsonable(value: Any) -> Any:
    """Last-resort coercion for things `json` refuses.

    numpy scalars and 0-d arrays, torch tensors and `Path` are what actually
    turn up in a training loop. Anything else becomes its `repr` rather than
    raising, because losing a whole run's log to one unserialisable field in one
    record is a worse failure than one ugly string.
    """
    for attr in ("item", "tolist"):  # numpy scalar / 0-d array, torch tensor
        fn = getattr(value, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:
                pass
    if isinstance(value, Path):
        # as_posix, not str: everything here runs on Windows, and str() would
        # put backslashes in a log that is meant to be read anywhere.
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
    """A run-scoped jsonl, plus a `meta.json` that makes the directory readable.

    The directory is named by run id and not by a config hash, for the reason
    the architecture doc settled: two runs at identical config and different
    seeds share a hash and produce different results, so a hash-named directory
    would have them overwrite each other. The hash lives *in* the records.

    Creating a run whose directory already exists raises rather than appending.
    Two processes interleaving lines into one metrics file is the same class of
    bug as two generation runs writing one shard directory, and it is silent.
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

        # Imported here, not at module scope: the flag is the whole point, and a
        # top-level `import wandb` would make an optional viewer a hard
        # dependency of every training run.
        #
        # Verified 2026-08-29 against wandb 0.29.0 by the offline branch of
        # `_self_check`: this init, three `log` calls and `finish` all run, and
        # the jsonl path is unaffected. The credential failures are verified
        # 2026-08-30 and all three land here, at init, before step 0 - see the
        # `wandb.login` note below. **The upload is verified 2026-08-30 too**,
        # by `--network` below: a three-record run reached the server, and its
        # history read back through `wandb.Api()` matched what was logged while
        # the jsonl underneath stayed intact. Nothing about the mirror is
        # unverified now.
        self._wandb = None
        if wandb_project is not None:
            import wandb  # noqa: PLC0415

            # `wandb.init` authenticates implicitly, and with no key configured
            # that means an **interactive prompt with no timeout**: a run
            # launched from a terminal sits at "Enter your choice:" forever,
            # having logged nothing. That is the one credential outcome worse
            # than crashing, and it is the default. `Settings(login_timeout=)`
            # does not reach the prompt - only an explicit `login(timeout=)`
            # does. Measured 2026-08-30 against 0.29.0; see the verification log.
            #
            # `login` returning False is the other quiet case: the prompt timed
            # out and W&B disabled itself, so `init` would then succeed and
            # mirror nothing at all. Raise instead. A mirror that is silently
            # off is worse than a run that refuses to start, because the run it
            # was meant to record is the multi-hour one.
            #
            # Offline and disabled runs are exempt on purpose: neither contacts
            # the server, so neither needs a key, and the offline self-check
            # below is exactly that case. Known consequence, accepted: `login`
            # verifies against the server and wandb reports unreachability as
            # an auth error, so a transient outage also stops the run here,
            # labelled auth. The mirror is opt-in and the jsonl needs no
            # network, so re-running without `wandb_project` is the way out.
            if wandb.setup().settings.mode == "online" and not wandb.login(
                timeout=WANDB_LOGIN_TIMEOUT_S
            ):
                raise RuntimeError(
                    "W&B is online but no API key is configured, and the login "
                    f"prompt was not answered within {WANDB_LOGIN_TIMEOUT_S}s. "
                    "Set WANDB_API_KEY, or run `wandb login`, or set "
                    "WANDB_MODE=offline - never a key in a config or the repo."
                )

            # x_disable_stats kills the background system-metrics sampler. It
            # samples CPU/GPU/disk on its own schedule, and sampling the GPU
            # during a tail-latency measurement inflates p99 quietly.
            self._wandb = wandb.init(
                project=wandb_project, name=self.run_id,
                config=dict(config or {}) | self.hashes,
                settings=wandb.Settings(x_disable_stats=True),
            )

    def log(self, record: Mapping[str, Any]) -> dict:
        """Append one record. Returns the line as written, parsed back.

        Returning the round-trip and not the input dict is deliberate. The
        caller passes numpy scalars and arrays, which `_jsonable` coerces during
        `dumps` - so the dict handed in and the line on disk are *not* the same
        object, and a caller that asserts against the input is asserting against
        something that was never written. The self-check caught exactly that.

        Flushed per line, not buffered: a run that dies at epoch 12 should still
        have epochs 1-11 on disk, and that is worth more than the syscalls.
        `t` is seconds since the run started, so a record is readable without
        knowing when the run began.
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

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._wandb is not None:
            self._wandb.finish()
            self._wandb = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def _self_check() -> None:
    """The plan's working-when, against a throwaway directory.

    W&B is not exercised. It is absent from this environment, which is exactly
    the condition this file has to survive - so the jsonl path is verified and
    the mirror is not.
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

        # A run must not reopen a directory that already holds one, or two
        # processes interleave lines into one file and neither says so.
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            raise AssertionError("re-creating an existing run directory did not raise")
        except FileExistsError:
            print("a second run cannot reopen an existing run directory")

        # The plan names pandas explicitly, so it is checked and not assumed.
        df = pd.read_json(path, lines=True)
        assert len(df) == 5 and df["step"].tolist() == list(range(5))
        assert set(hashes) <= set(df.columns), "pandas lost the hash columns"
        print(f"pandas.read_json(lines=True): {df.shape[0]} rows x {df.shape[1]} columns, "
              f"columns {sorted(df.columns)}")

        # Coercion, on the things a training loop actually produces.
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

    # The W&B mirror, exercised end to end in **offline** mode. Offline is not a
    # weaker check of the thing that was actually at risk: the risk was never the
    # network, it was that `wandb.init(...)`/`Settings(x_disable_stats=...)` might
    # not match the installed version's signature and would blow up at run start,
    # killing a multi-hour training run at minute zero. Offline runs the same
    # constructor and the same `log`/`finish` calls against a local directory.
    # What offline does NOT check is upload, auth, or the server-side view.
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        print("W&B mirror NOT exercised - wandb is absent from this environment")
    else:
        prev = os.environ.get("WANDB_MODE")
        os.environ["WANDB_MODE"] = "offline"
        try:
            # ignore_cleanup_errors, and not by preference: on Windows wandb
            # still holds `wandb/offline-run-*/logs/debug-internal.log` open
            # after `finish()` returns, so the rmtree raises WinError 32 and
            # fails a self-check whose subject already passed. Verified 0.29.0.
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

        # teardown, and not for tidiness: `wandb.setup()` caches a
        # process-global session on first use and then **ignores** later
        # WANDB_MODE changes - it says so, as a warning. Without this the next
        # check inherits `offline` from the block above, skips the online guard
        # it exists to test, and passes for the wrong reason. It did.
        wandb.teardown()
        _bad_credentials_check()

    print("logging self-check ok")


def _bad_credentials_check() -> None:
    """A wrong key must fail at construction, before a single record.

    The credential failure this guards against is not the crash - it is the
    *quiet* outcome, where the mirror turns itself off and the run trains for
    hours reporting to nothing. Both wrong-key and no-key land here, at `Run`
    construction; the no-key-on-a-terminal case is the one that used to hang,
    and it cannot be exercised without a terminal, so this checks the one that
    can be.

    **What this cannot tell you:** whether the credential was what failed.
    wandb raises `AuthenticationError` both when the server rejects a key and
    when the server cannot be reached - `_verify_login`'s own docstring says
    "rejects the credentials or cannot be reached", and it converts connection
    failures to that type deliberately, to keep its contract. So on a machine
    with no network this check sees the same exception type it sees on a
    rejection. Narrowing the `except` to `AuthenticationError`/`UsageError` was
    tried and abandoned for exactly that reason; do not re-attempt it, and do
    not discriminate on the message text either - a check that silently breaks
    when wandb rephrases a string is worse than one that reads honestly.

    What it therefore proves: `Run` refuses to construct, before step 0, rather
    than starting a mirror that is silently off. What it does not prove: that
    the refusal was caused by the credential.
    """
    import tempfile

    prev_key, prev_mode = os.environ.get("WANDB_API_KEY"), os.environ.get("WANDB_MODE")
    os.environ["WANDB_API_KEY"] = "0" * 40  # syntactically valid, no such account
    os.environ.pop("WANDB_MODE", None)
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            os.environ["WANDB_DIR"] = tmp
            try:
                Run("badkey", root=Path(tmp), wandb_project="mirage-selfcheck")
            except Exception as exc:  # noqa: BLE001 - see the docstring
                print(f"bad credentials: Run() refused to construct, before "
                      f"step 0, with {type(exc).__name__} - a type consistent "
                      f"with either a rejected key or an unreachable server")
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


def _network_check(project: str) -> None:
    """The one thing offline cannot check: a real run, uploaded and read back.

    Deliberately tiny - three records, no GPU. Reading the history back through
    the public API is the point: `finish()` returning is not evidence that
    anything arrived, only that nothing raised on the way out.
    """
    import tempfile

    import wandb

    assert os.environ.get("WANDB_MODE") in (None, "online"), "WANDB_MODE is not online"
    records = [{"step": i, "loss": 1.0 / (i + 1)} for i in range(3)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["WANDB_DIR"] = tmp
        try:
            with Run("netcheck", {"h": "0" * 8}, config={"epochs": 1},
                     root=Path(tmp), wandb_project=project) as run:
                assert run._wandb is not None, "wandb_project was passed but no run was made"
                path, url, wid, entity = run.path, run._wandb.url, run._wandb.id, run._wandb.entity
                print(f"authenticated as {entity!r}, run {url}")
                for r in records:
                    run.log(r)
        finally:
            os.environ.pop("WANDB_DIR", None)

        # The local log is authoritative and must be untouched by the mirror.
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").strip().splitlines()]
        assert len(rows) == len(records), f"{len(rows)} local lines for {len(records)} logs"
        for row, rec in zip(rows, records):
            assert {k: row[k] for k in rec} == rec, "a local record does not match what was logged"
        print(f"local jsonl intact underneath the mirror: {len(rows)} records, "
              f"each carrying run_id and h")

    # Server-side, through the public API - a different process's view.
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
    import sys

    # `--network <project>` needs a real key, from the environment or `wandb
    # login` - never from a file in this repo. Everything else runs without one.
    if len(sys.argv) > 2 and sys.argv[1] == "--network":
        _network_check(sys.argv[2])
    else:
        _self_check()
