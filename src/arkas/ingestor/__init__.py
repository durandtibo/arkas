r"""Contain the data ingestor."""

from __future__ import annotations

__all__ = ["BaseIngestor", "setup_ingestor", "is_ingestor_config"]

from arkas.ingestor.base import BaseIngestor, is_ingestor_config, setup_ingestor
