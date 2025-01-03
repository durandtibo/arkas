r"""Contain the implementation of a figure exporter."""

from __future__ import annotations

__all__ = ["FigureExporter"]

import logging
from typing import TYPE_CHECKING

from coola.utils import str_indent, str_mapping
from coola.utils.path import sanitize_path
from iden.io import BaseSaver, PickleSaver, setup_saver

from arkas.exporter.base import BaseExporter

if TYPE_CHECKING:
    from pathlib import Path

    from arkas.output.base import BaseOutput

logger = logging.getLogger(__name__)


class FigureExporter(BaseExporter):
    r"""Implement a simple figure exporter.

    Args:
        path: The path where to save the figures.
        saver: The figure saver or its configuration.
        exist_ok: If ``exist_ok`` is ``False`` (the default),
            an exception is raised if the path already exists.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> from pathlib import Path
    >>> import numpy as np
    >>> from arkas.output import AccuracyOutput
    >>> from arkas.state import AccuracyState
    >>> from arkas.exporter import FigureExporter
    >>> output = AccuracyOutput(
    ...     state=AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     exporter = FigureExporter(path=Path(tmpdir).joinpath("figures.pkl"))
    ...     exporter.export(output)
    ...

    ```
    """

    def __init__(
        self,
        path: Path | str,
        saver: BaseSaver | dict | None = None,
        exist_ok: bool = False,
    ) -> None:
        self._path = sanitize_path(path)
        self._saver = setup_saver(saver or PickleSaver())
        self._exist_ok = exist_ok

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "path": self._path,
                    "saver": self._saver,
                    "exist_ok": self._exist_ok,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def export(self, output: BaseOutput) -> None:
        logger.info("Exporting figures...")
        logger.info("Computing figures...")
        figures = output.get_plotter().plot()
        logger.info(f"Saving figures at {self._path}...")
        self._saver.save(figures, path=self._path, exist_ok=self._exist_ok)
