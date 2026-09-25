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
                            "offpalette_tau", "offpalette_frac_max",
                            "continuity_link_angle_max", "continuity_link_major_max",
                            "continuity_link_minor_max", "continuity_block_centre_max"]),
    "tokenizer": frozenset(["codebook_size", "stride"]),
    # Every knob that shapes the model, so two checkpoints that differ in one
    # never log the same `dynamics_hash`. Training knobs stay out: they travel in
    # the checkpoint's `knobs` dict. So does anything derived from upstream (the
    # vocabulary, tokens per frame, `data.ctx`), which an upstream hash already names.
    "dynamics": frozenset(["d_model", "n_layers", "n_heads", "mlp_ratio",
                           "pos_encoding", "output_head", "mask", "rope_base"]),
    "engine": frozenset(),
}

# Counts that must be a positive int. A zero or negative value still hashes
# fine, but names a run that cannot exist, so the hash no longer identifies
# anything real.
POSITIVE_INT_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["episodes", "steps_per_episode", "height", "width", "frames_per_shard",
                      "action_hold_steps"]),
    "data": frozenset(["ctx"]),
    "tokenizer": frozenset(["codebook_size", "stride"]),
    "dynamics": frozenset(["d_model", "n_layers", "n_heads", "mlp_ratio"]),
}

# Model choices taken by decision, each allowed only the value decided
# (`docs/phase2_structural_plan.md`, "Decisions"): RoPE and the untied output
# head by decisions 2 and 3, the block-causal mask by decision 9's measurement.
# The mask also admits `strict_causal`, the arm that measurement rejected, so
# `bench/mask_probe.py`'s strict arm runs from a config that names it. Any other
# alternative is an edit here, made when that decision's recorded trigger fires,
# never a JSON edit alone. A misspelling would otherwise hash as a model that
# differs from the decided one in name only.
CHOICE_KEYS: dict[str, dict[str, frozenset[str]]] = {
    "dynamics": {
        "pos_encoding": frozenset(["rope"]),
        "output_head": frozenset(["untied"]),
        "mask": frozenset(["block_causal", "strict_causal"]),
    },
}

# Fractions of a frame used as pass/fail thresholds. Zero is allowed: "no pixel
# off the palette" is the right bar for real renders, which always meet it. Only
# decoder output can never reach zero, and the schema should still be able to
# express the render setting so the two stay comparable.
#
# `offpalette_frac_max` is a fraction of the frame's pixels rather than a pixel
# count, because a count needs a separately calibrated value per resolution. A
# percentile of colour distance was tried first and failed; see
# `validator._weighted_pctl`.
#
# Not in FRACTION_KEYS even though it is a fraction: those must be below 1.0, and
# exactly 1.0 ("no frame can ever fail this") is a reasonable value while a
# threshold is still being calibrated.
NON_NEGATIVE_FLOAT_KEYS: dict[str, frozenset[str]] = {
    "validator": frozenset(["offpalette_frac_max"]),
}

# The coherence horizon's continuity bounds (gate row 4 of Phase 2): the
# largest per-step change each object's feature may make, one value per link or
# block in the palette's name order - radians for an angle, pixels otherwise.
# Lists, because the object count comes from the scene XML, which this file
# does not read; `mirage.dynamics_eval.continuity_bounds` checks the lengths
# against the palette. Zero is allowed for the same reason as above.
NON_NEGATIVE_FLOAT_LIST_KEYS: dict[str, frozenset[str]] = {
    "validator": frozenset(["continuity_link_angle_max", "continuity_link_major_max",
                            "continuity_link_minor_max", "continuity_block_centre_max"]),
}

# Rates and splits, all of which must lie in [0, 1).
FRACTION_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["reach_digit_noise_prob"]),
    "data": frozenset(["val_fraction"]),
    "validator": frozenset(["contact_rate_min", "recoverable_occlusion_rate_min"]),
}

# Physical thresholds, in metres or metres per radian. Positive with no upper
# bound, which is why they are not FRACTION_KEYS: a distance over 1 m becomes
# valid as soon as the scene grows.
#
# Strictly positive, even jacobian_deadband, where zero might seem reasonable: a
# deadband of 0 behaves the same as 1e-9, so "off" can be written as a tiny
# positive number and the rule stays simple.
POSITIVE_FLOAT_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["reach_done_dist", "jacobian_deadband"]),
    # A distance in RGB space, so its maximum is sqrt(3) * 255 = 441.7, not 1.
    "validator": frozenset(["offpalette_tau"]),
    "dynamics": frozenset(["rope_base"]),
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
    # `type(v) is not int` rather than isinstance: bool is a subclass of int, so
    # isinstance would accept `true` as the positive integer 1.
    for section, keys in POSITIVE_INT_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if type(value) is not int or value <= 0:
                raise ValueError(f"{section}.{key} must be a positive int, got {value!r}")

    for section, keys in NON_NEGATIVE_FLOAT_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if type(value) not in (int, float) or value < 0.0:
                raise ValueError(f"{section}.{key} must be a non-negative float, got {value!r}")

    for section, keys in NON_NEGATIVE_FLOAT_LIST_KEYS.items():
        for key in sorted(keys):
            value = raw[section][key]
            if (type(value) is not list or not value
                    or any(type(v) not in (int, float) or v < 0.0 for v in value)):
                raise ValueError(f"{section}.{key} must be a non-empty list of non-negative "
                                 f"floats, got {value!r}")

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

    for section, choices in CHOICE_KEYS.items():
        for key, allowed in sorted(choices.items()):
            value = raw[section][key]
            if type(value) is not str or value not in allowed:
                raise ValueError(f"{section}.{key} must be one of {sorted(allowed)}, got {value!r}")

    seed = raw["sim"]["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError(f"sim.seed must be a non-negative int, got {seed!r}")

    # Without this, height // stride silently rounds down: a stride of 6 on 64 px
    # drops 4 px from every row, and it only shows up hours later as unexplained
    # tokenizer reconstruction error.
    stride = raw["tokenizer"]["stride"]
    for dim in ("height", "width"):
        size = raw["sim"][dim]
        if size % stride:
            raise ValueError(
                f"sim.{dim} ({size}) must be divisible by tokenizer.stride ({stride})"
            )


def scene_bytes(path: Path | str) -> bytes:
    """The scene XML bytes that go into `data_hash`, with line endings normalised.

    The raw XML bytes are part of `data_hash`, so a checkout with Windows (CRLF)
    line endings would hash the same scene differently from one with LF. The
    `eol=lf` rule in `.gitattributes` was meant to prevent that and did not: one
    working tree and a fresh clone of the same commit gave different hashes,
    identical once CR was stripped. Git applies `eol` only at checkout and never
    rewrites files already on disk.

    Normalising here works whatever the checkout did. The `.gitattributes` rule
    stays because it keeps diffs clean, but the hash no longer depends on it.
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

    # The order of the parts (sim, data, xml) is part of the hash's definition. Do not reorder.
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

    # Read-only views. frozen=True only stops reassigning the fields. Without this
    # a caller could edit cfg.sim in place and end up with a config its data_hash no
    # longer describes, which is exactly what the hashes exist to prevent.
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
    """Smallest check that fails if the hash chain or the value checks break."""
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

    # Checks the *order* of the hash inputs, which the relative checks below
    # cannot: reordering the inputs in load() shifts all five hashes together, so
    # they still compare correctly with each other. Recomputing the documented order
    # here is the only thing that catches it. Not a fixed hash value, because the
    # scene XML is allowed to change and that must not fail this check.
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

    # The line-ending split, tested rather than trusted: a CRLF copy of the scene
    # must hash the same as the LF original. This would have caught the real split
    # the day it appeared.
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

    # A tokenizer or dynamics change must change every hash downstream of it and
    # none upstream. That is what the input order in load() guarantees. The
    # dynamics knob is the head count, which sat in no hash before it moved here.
    tok = variant("tokenizer", "codebook_size", 1024)
    assert tok.data_hash == cfg.data_hash
    assert tok.validator_hash == cfg.validator_hash
    assert tok.tokenizer_hash != cfg.tokenizer_hash
    assert tok.dynamics_hash != cfg.dynamics_hash
    assert tok.engine_hash != cfg.engine_hash
    dyn = variant("dynamics", "n_heads", 8)
    assert dyn.data_hash == cfg.data_hash
    assert dyn.validator_hash == cfg.validator_hash
    assert dyn.tokenizer_hash == cfg.tokenizer_hash
    assert dyn.dynamics_hash != cfg.dynamics_hash
    assert dyn.engine_hash != cfg.engine_hash
    assert variant("dynamics", "mask", "strict_causal").dynamics_hash != cfg.dynamics_hash
    assert variant("dynamics", "rope_base", 500.0).dynamics_hash != cfg.dynamics_hash

    # A continuity bound is a validator threshold: it moves validator_hash and
    # nothing else, so recalibrating it orphans no dataset or checkpoint.
    cont = variant("validator", "continuity_block_centre_max", [1.0, 2.0, 3.0])
    assert cont.validator_hash != cfg.validator_hash
    assert (cont.data_hash, cont.tokenizer_hash, cont.dynamics_hash) == \
        (cfg.data_hash, cfg.tokenizer_hash, cfg.dynamics_hash)

    # A sim change must change every hash, on both branches.
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
        ("validator", "offpalette_frac_max", -1.0, "non-negative float"),
        ("validator", "offpalette_frac_max", True, "non-negative float"),
        ("validator", "continuity_link_angle_max", 0.5, "list of non-negative"),
        ("validator", "continuity_link_major_max", [], "list of non-negative"),
        ("validator", "continuity_block_centre_max", [1.0, -1.0, 1.0], "list of non-negative"),
        ("validator", "continuity_link_minor_max", [True, 1.0], "list of non-negative"),
        ("validator", "continuity_link_angle_max", _DROP, "missing keys"),
        ("sim", "reach_done_dist", 0.0, "positive float"),
        ("sim", "seed", _DROP, "missing keys"),
        ("sim", "extra", 1, "unknown keys"),
        ("dynamics", "n_heads", 0, "positive int"),
        ("dynamics", "mlp_ratio", True, "positive int"),
        ("dynamics", "mask", "causal", "must be one of"),
        ("dynamics", "pos_encoding", "learned", "must be one of"),
        ("dynamics", "output_head", ["untied"], "must be one of"),
        ("dynamics", "n_heads", _DROP, "missing keys"),
        ("dynamics", "extra", 1, "unknown keys"),
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
