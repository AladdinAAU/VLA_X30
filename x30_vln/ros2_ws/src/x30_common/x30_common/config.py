"""Loader for `configs/models.yaml` with environment overrides.

Schema (all keys optional; defaults below):

    perception:   {url, timeout_s}
    mapper_vlm:   {base_url, model, api_key, api_key_env, max_tokens, temperature, timeout_s, max_retries,
                   backoff_s, context_tokens, extra_body}
    reasoner_vlm: {same as mapper_vlm}

Override order (later wins): built-in defaults, `models.yaml`, environment. Environment names are
`X30_<ROLE>_<FIELD>` with ROLE in PERCEPTION, MAPPER, REASONER, for example `X30_REASONER_URL`,
`X30_REASONER_MODEL`, `X30_MAPPER_MAX_TOKENS`, `X30_PERCEPTION_URL`. `URL` is an alias of `base_url`.
`X30_MODELS_CONFIG` points at another yaml file; `X30_CONFIG_DIR` (or `X30_CONFIGS`, set by compose) at the `configs/` folder.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULTS: dict[str, dict] = {
    "perception": {"url": "http://perception:8100", "timeout_s": 10.0},
    "mapper_vlm": {"base_url": "http://vlm:8000/v1", "model": "qwen3-vl", "api_key": "none", "api_key_env": "",
                   "max_tokens": 512, "temperature": 0.2, "timeout_s": 60.0, "max_retries": 3, "backoff_s": 1.0,
                   "context_tokens": 8192, "extra_body": {}},
    "reasoner_vlm": {"base_url": "http://vlm:8000/v1", "model": "qwen3-vl", "api_key": "none", "api_key_env": "",
                     "max_tokens": 400, "temperature": 0.2, "timeout_s": 120.0, "max_retries": 3, "backoff_s": 1.0,
                     "context_tokens": 8192, "extra_body": {}},
}
ROLE_ENV = {"perception": "PERCEPTION", "mapper_vlm": "MAPPER", "reasoner_vlm": "REASONER"}
ROLE_ALIASES = {"perception": "perception", "mapper": "mapper_vlm", "mapper_vlm": "mapper_vlm",
                "reasoner": "reasoner_vlm", "reasoner_vlm": "reasoner_vlm"}


def find_config_dir(start: str | os.PathLike | None = None) -> Path | None:
    """`configs/` directory: $X30_CONFIG_DIR, else the nearest `configs/` with a models.yaml above `start`, cwd or this file."""
    for name in ("X30_CONFIG_DIR", "X30_CONFIGS"):          # X30_CONFIGS is set by docker/compose.yml
        env = os.environ.get(name)
        if env and Path(env).is_dir():
            return Path(env)
    for base in (start, Path.cwd(), Path(__file__).resolve()):
        if base is None:
            continue
        p = Path(base).resolve()
        for cand in [p, *p.parents]:
            if (cand / "configs" / "models.yaml").is_file():
                return cand / "configs"
    return None


def _cast(value: str, default: Any) -> Any:
    if isinstance(default, bool):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    if isinstance(default, (dict, list)):
        return json.loads(value)
    return value


@dataclass
class VLMConfig:
    base_url: str = "http://vlm:8000/v1"
    model: str = "qwen3-vl"
    api_key: str = "none"
    max_tokens: int = 400
    temperature: float = 0.2
    timeout_s: float = 120.0
    max_retries: int = 3
    backoff_s: float = 1.0
    context_tokens: int = 8192
    extra_body: dict = field(default_factory=dict)

    def as_dict(self, redact: bool = True) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        if redact and d["api_key"] not in ("", "none"):
            d["api_key"] = "***"
        return d


@dataclass
class PerceptionConfig:
    url: str = "http://perception:8100"
    timeout_s: float = 10.0


@dataclass
class ModelsConfig:
    perception: PerceptionConfig
    mapper_vlm: VLMConfig
    reasoner_vlm: VLMConfig
    source: str = "defaults"

    def vlm(self, role: str) -> VLMConfig:
        key = ROLE_ALIASES.get(role)
        if key not in ("mapper_vlm", "reasoner_vlm"):
            raise KeyError(f"unknown VLM role {role!r}; use 'mapper' or 'reasoner'")
        return getattr(self, key)

    def as_dict(self) -> dict:
        return {"perception": vars(self.perception).copy(), "mapper_vlm": self.mapper_vlm.as_dict(),
                "reasoner_vlm": self.reasoner_vlm.as_dict(), "source": self.source}

    def hash(self) -> str:
        return stable_hash(self.as_dict())


def load_models_config(path: str | os.PathLike | None = None, env: Mapping[str, str] | None = None) -> ModelsConfig:
    env = os.environ if env is None else env
    merged = copy.deepcopy(DEFAULTS)
    src = "defaults"
    file_path = path or env.get("X30_MODELS_CONFIG")
    if not file_path:
        d = find_config_dir()
        file_path = (d / "models.yaml") if d else None
    if file_path and Path(file_path).is_file():
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for role, vals in data.items():
            key = ROLE_ALIASES.get(role, role)
            if key in merged and isinstance(vals, dict):
                merged[key].update(vals)
        src = str(file_path)
    for role, prefix in ROLE_ENV.items():
        for key, default in list(merged[role].items()):
            names = [f"X30_{prefix}_{key.upper()}"]
            if key in ("base_url", "url"):
                names.append(f"X30_{prefix}_URL")
            for n in names:
                if n in env and env[n] != "":
                    merged[role][key] = _cast(env[n], default)
                    src += f" +env:{n}"
                    break
    for role in ("mapper_vlm", "reasoner_vlm"):
        m = merged[role]
        if m.get("api_key_env") and env.get(m["api_key_env"]):
            m["api_key"] = env[m["api_key_env"]]
        if isinstance(m.get("extra_body"), str):
            m["extra_body"] = json.loads(m["extra_body"] or "{}")
    vlm_keys = {f.name for f in fields(VLMConfig)}
    return ModelsConfig(
        perception=PerceptionConfig(**{k: v for k, v in merged["perception"].items() if k in ("url", "timeout_s")}),
        mapper_vlm=VLMConfig(**{k: v for k, v in merged["mapper_vlm"].items() if k in vlm_keys}),
        reasoner_vlm=VLMConfig(**{k: v for k, v in merged["reasoner_vlm"].items() if k in vlm_keys}),
        source=src)


def load_yaml(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stable_hash(obj: Any, n: int = 12) -> str:
    """Short deterministic hash of JSON-able data (dicts sorted)."""
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


def file_hash(*paths: str | os.PathLike, n: int = 12) -> str:
    """Hash of the contents of several files (order matters). Missing files hash as their name only."""
    h = hashlib.sha256()
    for p in paths:
        h.update(str(Path(p).name).encode())
        try:
            h.update(Path(p).read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()[:n]
