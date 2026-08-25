
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple
import hashlib

EXPECTED_KEYS: dict[str, frozenset[str]] = {
    "sim": frozenset(["scene_xml", "seed", "episodes", "steps_per_episode", "height", "width", "frames_per_shard"]),
    "data": frozenset(["shard_dir", "ctx", "val_fraction"]),
    "validator": frozenset(["contact_rate_min", "occlusion_rate_min"]),
    "tokenizer": frozenset(["codebook_size", "stride"]),
    "dynamics": frozenset(["d_model", "n_layers"]),
    "engine": frozenset(),
}


class Shapes(NamedTuple):
    image_size: tuple[int, int]
    token_grid: tuple[int, int]
    context_length: int


@dataclass(frozen=True)
class Config:
    sim: dict[str, Any]
    data: dict[str, Any]
    validator: dict[str, Any]
    tokenizer: dict[str, Any]
    dynamics: dict[str, Any]
    engine: dict[str, Any]
    data_hash: str
    tokenizer_hash: str
    dynamics_hash: str
    engine_hash: str
    validator_hash: str
    shapes: Shapes


def _check_keys(label: str, expected: frozenset[str], actual: set[str]) -> None:
    expected = frozenset(expected)
    actual = set(actual)

    missing = expected - actual
    unknown = actual - expected

    if missing:
        raise ValueError(f"{label} config missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} config unknown keys: {sorted(unknown)}")


def load(path) -> Config:

    def canon(section) -> bytes:
        return json.dumps(section, sort_keys=True, separators=(",", ":")).encode()
    
    with open(path, "r") as f:
        raw = json.load(f)
    

    _check_keys("top-level", frozenset(EXPECTED_KEYS.keys()), set(raw.keys()))
    for section, expected_keys in EXPECTED_KEYS.items():
        _check_keys(section, expected_keys, set(raw[section].keys()))
    repo_root = Path(__file__).resolve().parent.parent
    scene = repo_root / raw["sim"]["scene_xml"]

    xml_bytes = scene.read_bytes()

    # data_hash term order (sim, data, xml) is part of the definition - do not reorder
    data_hash = hashlib.sha256(canon(raw["sim"]) + canon(raw["data"]) + xml_bytes).hexdigest()
    tokenizer_hash = hashlib.sha256(data_hash.encode() + canon(raw["tokenizer"])).hexdigest()
    dynamics_hash = hashlib.sha256(tokenizer_hash.encode() + canon(raw["dynamics"])).hexdigest()
    engine_hash = hashlib.sha256(dynamics_hash.encode() + canon(raw["engine"])).hexdigest()
    validator_hash = hashlib.sha256(data_hash.encode() + canon(raw["validator"])).hexdigest()

    shapes = Shapes(
        image_size=(raw["sim"]["height"], raw["sim"]["width"]),
        token_grid=(raw["sim"]["height"] // raw["tokenizer"]["stride"],
                    raw["sim"]["width"] // raw["tokenizer"]["stride"]),
        context_length=raw["data"]["ctx"],
    )

    return Config(
        **raw,
        data_hash=data_hash,
        tokenizer_hash=tokenizer_hash,
        dynamics_hash=dynamics_hash,
        engine_hash=engine_hash,
        validator_hash=validator_hash,
        shapes=shapes,
    )









