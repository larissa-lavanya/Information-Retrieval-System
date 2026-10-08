"""
STORAGE LAYER (backend-agnostic persistence)
--------------------------------------------
``StorageAdapter`` is the single interface the rest of the system uses to
persist and restore the index.  Swapping backends only means changing
``storage.backend`` in config.yaml:

    memory - keeps the serialised index in a Python dict (lost on exit)
    json   - writes it to a JSON file on disk

New backends (a key-value store, a relational database, ...) only need to
implement ``save`` / ``load`` / ``exists``.
"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from src.config import project_path


class StorageAdapter(ABC):
    @abstractmethod
    def save(self, state: Dict[str, Any]) -> None: ...

    @abstractmethod
    def load(self) -> Optional[Dict[str, Any]]: ...

    def exists(self) -> bool:
        return self.load() is not None


class MemoryStorage(StorageAdapter):
    """Process-local RAM storage."""

    def __init__(self) -> None:
        self._state: Optional[Dict[str, Any]] = None

    def save(self, state: Dict[str, Any]) -> None:
        self._state = state

    def load(self) -> Optional[Dict[str, Any]]:
        return self._state


class JsonFileStorage(StorageAdapter):
    """Persist the index as a JSON file."""

    def __init__(self, path: str) -> None:
        self.path = path if os.path.isabs(path) else project_path(path)

    def save(self, state: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle)
        os.replace(tmp, self.path)  # atomic swap so a crash never leaves a half-written file

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return None
        with open(self.path, "r", encoding="utf-8") as handle:
            return json.load(handle)


_memory_singleton = MemoryStorage()


def create_storage(config: Dict[str, Any]) -> StorageAdapter:
    """Factory: pick the backend named in ``config['storage']['backend']``."""
    storage_cfg = config["storage"]
    backend = storage_cfg["backend"]
    if backend == "memory":
        return _memory_singleton
    if backend == "json":
        return JsonFileStorage(storage_cfg["json_path"])
    raise ValueError(f"Unknown storage backend: {backend!r} (use 'memory' or 'json')")
