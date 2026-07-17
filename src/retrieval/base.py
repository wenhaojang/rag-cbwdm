"""Common interface for retrieval backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RetrievalBackend(ABC):
    @abstractmethod
    def search(self, query: str, top_n: int) -> list[dict[str, Any]]:
        """Return ranked documents with stored fields."""
