"""Config loader — resolves model aliases from config/default.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    """Load the YAML config as a plain dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_model_id(cfg: dict, kind: str, alias: str | None = None) -> tuple[str, str]:
    """Resolve a draft/target alias to its HF repo id.

    kind is "draft" or "target". Returns (alias, repo_id). The active alias
    defaults to cfg["model"][kind] — the one-line switch (§2 drop-in).
    """
    alias = alias or cfg["model"][kind]
    catalog = cfg["models"][kind + "s"]
    if alias not in catalog:
        raise KeyError(f"unknown {kind} alias {alias!r}; available: {sorted(catalog)}")
    return alias, catalog[alias]
