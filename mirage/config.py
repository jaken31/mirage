import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NamedTuple

EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["scene_xml", "seed", "episodes", "steps_per_episode", "height", "width", "frames_per_shard",
                      "action_hold_steps", "reach_digit_noise_prob", "jacobian_deadband",
                      "reach_done_dist"]),
    "data": frozenset(["shard_dir", "ctx", "val_fraction"]),
    "validator": frozenset(["contact_rate_min", "recoverable_occlusion_rate_min",
                            "offpalette_tau"]),
    "tokenizer": frozenset(["codebook_size", "stride"]),
    "dynamics": frozenset(["d_model", "n_layers"]),
    "engine": frozenset(),
}

# Counts that must be a positive int. Zero or negative hashes perfectly well and
# names a run that cannot exist, so the hash stops identifying a real artifact.
POSITIVE_INT_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["episodes", "steps_per_episode", "height", "width", "frames_per_shard",
                      "action_hold_steps"]),
    "data": frozenset(["ctx"]),
    "tokenizer": frozenset(["codebook_size", "stride"]),
    "dynamics": frozenset(["d_model", "n_layers"]),
}

# Rates and splits, all of which must lie in [0, 1).
FRACTION_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["reach_digit_noise_prob"]),
    "data": frozenset(["val_fraction"]),
    "validator": frozenset(["contact_rate_min", "recoverable_occlusion_rate_min"]),
}

# Physical thresholds - metres, or metres per radian. Positive but unbounded
# above, which is why they are not FRACTION_KEYS: a distance over 1 m is
# legitimate the moment the scene grows, and validating one as a fraction would
# reject it for no reason.
#
# Positive rather than merely non-negative, including jacobian_deadband, where
# zero would be defensible: a deadband of 0 and one of 1e-9 behave identically,
# so "off" costs nothing to spell as a small positive number, and the bound stays
# one rule instead of two.
POSITIVE_FLOAT_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["reach_done_dist", "jacobian_deadband"]),
    # An RGB Euclidean radius, so its ceiling is sqrt(3) * 255 = 441.7 and not 1.
    "validator": frozenset(["offpalette_tau"]),
}


class Shapes(NamedTuple):
    image_size: tuple[int, int]
    token_grid: tuple[int, int]
    context_length: int


@dataclass(frozen=True)
class Config:
    sim: Mapping[str, Any]
    data: Mapping[str, Any]
    validator: Mapping[str, Any]
    tokenizer: Mapping[str, Any]
    dynamics: Mapping[str, Any]
    engine: Mapping[str, Any]
    data_hash: str
    tokenizer_hash: str
    dynamics_hash: str
    engine_hash: str
    validator_hash: str
    shapes: Shapes


def _check_keys(label: str, expected: frozenset[str], actual: set[str]) -> None:
    missing = expected - actual
    unknown = actual - expected

    if missing:
        raise ValueError(f"{label} config missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} config unknown keys: {sorted(unknown)}")


def _check_values(raw: dict[str, Any]) -> None:
    # `type(v) is not int` rather than isinstance: bool subclasses int, so
    # isinstance would accept `true` as the positive integer 1.
    for section, keys in POSITIVE_INT_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive int, got {value!r}")

    for section, keys in POSITIVE_FLOAT_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if type(value) not in (int, float) or value <= 0.0:
                raise ValueError(f"{section}.{key} must be a positive float, got {value!r}")

    for section, keys in FRACTION_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if type(value) not in (int, float) or not 0.0 <= value < 1.0:
                raise ValueError(f"{section}.{key} must be in [0, 1), got {value!r}")

    seed = raw["sim"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError(f"sim.seed must be a non-negative int, got {seed!r}")

    # Without this, height // stride truncates silently: a stride of 6 on 64 px
    # drops 4 px from every row and surfaces only as unexplained tokenizer
    # reconstruction error, hours downstream.
    stride = raw["tokenizer"]["stride"]
    for dim in ("height", "width"):
        size = raw["sim"][dim]
        if size % stride:
            raise ValueError(
                f"sim.{dim} ({size}) must be divisible by tokenizer.stride ({stride})"
            )


def scene_bytes(path: Path | str) -> bytes:
    """The scene XML as it enters `data_hash`, with line endings normalised.

    The XML's raw bytes are a term in `data_hash`, so a CRLF working tree hashes
    the same scene differently from an LF one. `.gitattributes` sets `eol=lf`
    precisely to stop that and **it did not**: measured 2026-08-28, this worktree
    read `219ab0af` while a fresh clone of the same commit read `18a76531` -
    byte-identical once CR is stripped. Git applies `eol` only at checkout and
    never rewrites a working tree that already exists, so the rule silently
    skipped every file that was on disk before the attribute landed.

    Normalising here does not depend on any checkout honouring an attribute,
    which is the difference between a convention and a guarantee. The
    `.gitattributes` rule stays - it keeps the *diffs* sane - but nothing about
    provenance rests on it now.
    """
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def _canon(section: Mapping[str, Any]) -> bytes:
    return json.dumps(section, sort_keys=True, separators=(",", ":")).encode()


def load(path: Path | str) -> Config:
    with open(path, "r") as f:
        raw = json.load(f)

    _check_keys("top-level", frozenset(EXPECTED_KEYS.keys()), set(raw.keys()))
    for section, expected_keys in EXPECTED_KEYS.items():
        _check_keys(section, expected_keys, set(raw[section].keys()))
    _check_values(raw)

    repo_root = Path(__file__).resolve().parent.parent
    scene = repo_root / raw["sim"]["scene_xml"]

    xml_bytes = scene_bytes(scene)

    # data_hash term order (sim, data, xml) is part of the definition - do not reorder
    data_hash = hashlib.sha256(_canon(raw["sim"]) + _canon(raw["data"]) + xml_bytes).hexdigest()
    tokenizer_hash = hashlib.sha256(data_hash.encode() + _canon(raw["tokenizer"])).hexdigest()
    dynamics_hash = hashlib.sha256(tokenizer_hash.encode() + _canon(raw["dynamics"])).hexdigest()
    engine_hash = hashlib.sha256(dynamics_hash.encode() + _canon(raw["engine"])).hexdigest()
    validator_hash = hashlib.sha256(data_hash.encode() + _canon(raw["validator"])).hexdigest()

    shapes = Shapes(
        image_size=(raw["sim"]["height"], raw["sim"]["width"]),
        token_grid=(raw["sim"]["height"] // raw["tokenizer"]["stride"],
                    raw["sim"]["width"] // raw["tokenizer"]["stride"]),
        context_length=raw["data"]["ctx"],
    )

    # Read-only views. frozen=True only stops rebinding the fields; without this
    # a caller could mutate cfg.sim and hold a config whose data_hash no longer
    # describes it, which is the exact failure the hash tree exists to prevent.
    sections = {name: MappingProxyType(raw[name]) for name in EXPECTED_KEYS}

    return Config(
        **sections,
        data_hash=data_hash,
        tokenizer_hash=tokenizer_hash,
        dynamics_hash=dynamics_hash,
        engine_hash=engine_hash,
        validator_hash=validator_hash,
        shapes=shapes,
    )


_DROP = object()


def _self_check() -> None:
    """Smallest check that fails if the hash chain or the validators break."""
    import copy
    import tempfile

    base_path = Path(__file__).resolve().parent / "configs" / "base.json"
    base_raw = json.loads(base_path.read_text())
    cfg = load(base_path)

    def variant(section: str, key: str, value: Any) -> Config:
        raw = copy.deepcopy(base_raw)
        if value is _DROP:
            del raw[section][key]
        else:
            raw[section][key] = value
        tmp = Path(tempfile.mkdtemp()) / "variant.json"
        tmp.write_text(json.dumps(raw))
        return load(tmp)

    assert cfg.shapes == Shapes((64, 64), (8, 8), 15), cfg.shapes

    # Pins the term *order*, which every relative check below misses: reordering
    # the terms in load() shifts all five hashes consistently, so they still
    # compare correctly against each other. Restating the documented order here
    # is the only thing that catches it. Not a pinned literal - the scene XML is
    # expected to change during Phase 0, and that must not fail this check.
    scene = Path(__file__).resolve().parent.parent / base_raw["sim"]["scene_xml"]
    xml_bytes = scene_bytes(scene)
    expect_data = hashlib.sha256(
        _canon(base_raw["sim"]) + _canon(base_raw["data"]) + xml_bytes).hexdigest()
    expect_tokenizer = hashlib.sha256(
        expect_data.encode() + _canon(base_raw["tokenizer"])).hexdigest()
    expect_validator = hashlib.sha256(
        expect_data.encode() + _canon(base_raw["validator"])).hexdigest()
    assert cfg.data_hash == expect_data, "data_hash term order changed"
    assert cfg.tokenizer_hash == expect_tokenizer, "tokenizer_hash term order changed"
    assert cfg.validator_hash == expect_validator, "validator_hash term order changed"

    # The CRLF fork, tested rather than trusted. A CRLF copy of the scene must
    # hash as its LF twin; without the normalisation this is the assertion that
    # would have caught the 219ab0af / 18a76531 split on the day it appeared.
    crlf = Path(tempfile.mkdtemp()) / "crlf.xml"
    crlf.write_bytes(xml_bytes.replace(b"\n", b"\r\n"))
    assert crlf.read_bytes() != xml_bytes, "the CRLF copy is identical - no newlines?"
    assert scene_bytes(crlf) == xml_bytes, "a CRLF scene does not hash as its LF twin"

    try:
        cfg.sim["seed"] = 999  # type: ignore[index]
    except TypeError:
        pass
    else:
        raise AssertionError("sim section is mutable")

    # A tokenizer change must invalidate everything downstream of it and nothing
    # upstream. This is what the term order in load() buys.
    tok = variant("tokenizer", "codebook_size", 1024)
    assert tok.data_hash == cfg.data_hash
    assert tok.validator_hash == cfg.validator_hash
    assert tok.tokenizer_hash != cfg.tokenizer_hash
    assert tok.dynamics_hash != cfg.dynamics_hash
    assert tok.engine_hash != cfg.engine_hash

    # A sim change must invalidate the whole tree, both branches.
    sim = variant("sim", "episodes", 2000)
    assert sim.data_hash != cfg.data_hash
    assert sim.validator_hash != cfg.validator_hash
    assert sim.tokenizer_hash != cfg.tokenizer_hash

    for section, key, value, msg in [
        ("tokenizer", "stride", 6, "divisible"),
        ("sim", "episodes", 0, "positive int"),
        ("sim", "episodes", True, "positive int"),
        ("sim", "seed", -1, "non-negative"),
        ("data", "val_fraction", 1.0, "[0, 1)"),
        ("sim", "action_hold_steps", 0, "positive int"),
        ("sim", "reach_digit_noise_prob", 1.0, "[0, 1)"),
        ("sim", "jacobian_deadband", 0.0, "positive float"),
        ("validator", "offpalette_tau", 0.0, "positive float"),
        ("sim", "reach_done_dist", 0.0, "positive float"),
        ("sim", "seed", _DROP, "missing keys"),
        ("sim", "extra", 1, "unknown keys"),
    ]:
        try:
            variant(section, key, value)
        except ValueError as e:
            assert msg in str(e), f"wrong error for {section}.{key}: {e}"
        else:
            raise AssertionError(f"expected ValueError mentioning {msg!r}")

    print("config self-check ok")


if __name__ == "__main__":
    _self_check()
