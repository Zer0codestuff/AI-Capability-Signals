"""Validated ingestion of public data sources.

Each module exposes a ``load_*`` function taking a :class:`aicap.netcache.Fetcher` and returning a
tidy frame that has passed a :class:`aicap.schema.Contract`. Ingestion normalises names and types
and nothing else: no scores, indices or filtering decisions are made here, so that every analytical
choice is visible in ``aicap.analysis``.
"""

from __future__ import annotations

from . import aei, epoch, lmarena, openrouter, swebench

__all__ = ["aei", "epoch", "lmarena", "openrouter", "swebench"]
