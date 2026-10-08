"""
Configuration layer.

Loads ``config.yaml`` and merges it over built-in defaults so the rest of the
code base can always rely on every key being present.  The result is a plain
nested ``dict`` that is cached and shared (a "globally accessible dict").
"""

import copy
import os
from typing import Any, Dict, Optional

DEFAULTS: Dict[str, Any] = {
    "dataset": {"source": "sample", "path": None, "max_docs": 1000},
    "storage": {"backend": "memory", "json_path": "data/index.json"},
    "preprocessing": {"remove_stopwords": True, "stemming": True},
    "ranking": {
        "scorer": "bm25",
        "top_k": 10,
        "bm25": {"k1": 1.5, "b": 0.75},
    },
    "spelling": {"enabled": True, "max_edit_distance": 2, "auto_correct": False},
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "config.yaml")
_cache: Optional[Dict[str, Any]] = None


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None, force_reload: bool = False) -> Dict[str, Any]:
    """Return the merged configuration dict (cached after the first call)."""
    global _cache
    if _cache is not None and path is None and not force_reload:
        return _cache

    config = copy.deepcopy(DEFAULTS)
    path = path or _DEFAULT_PATH
    if os.path.exists(path):
        try:
            import yaml  # optional dependency; defaults are used if missing

            with open(path, "r", encoding="utf-8") as handle:
                _deep_update(config, yaml.safe_load(handle) or {})
        except ImportError:
            print("[config] PyYAML not installed - using built-in defaults.")

    if path == _DEFAULT_PATH:
        _cache = config
    return config


def project_path(*parts: str) -> str:
    """Resolve a path relative to the project root."""
    return os.path.join(_PROJECT_ROOT, *parts)
