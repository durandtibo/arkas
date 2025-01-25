r"""Contain the implementation of a pairwise column co-occurrence
plotter."""

from __future__ import annotations

__all__ = ["BaseFigureCreator", "ColumnCooccurrencePlotter", "MatplotlibFigureCreator"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from arkas.figure.creator import FigureCreatorRegistry
from arkas.figure.html import HtmlFigure
from arkas.figure.matplotlib import MatplotlibFigure, MatplotlibFigureConfig
from arkas.figure.utils import MISSING_FIGURE_MESSAGE
from arkas.plot.utils import readable_xticklabels, readable_yticklabels
from arkas.plotter.caching import BaseStateCachedPlotter
from arkas.state.column_cooccurrence import ColumnCooccurrenceState

if TYPE_CHECKING:
    from arkas.figure.base import BaseFigure


class BaseFigureCreator(ABC):
    r"""Define the base class to create a figure of the pairwise column
    co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
    >>> from arkas.state import ColumnCooccurrenceState
    >>> creator = MatplotlibFigureCreator()
    >>> fig = creator.create(
    ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ... )

    ```
    """

    @abstractmethod
    def create(self, state: ColumnCooccurrenceState) -> BaseFigure:
        r"""Create a figure of the pairwise column co-occurrence matrix.

        Args:
            state: The state with the co-occurrence matrix.

        Returns:
            The generated figure.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.figure import MatplotlibFigureConfig
        >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
        >>> from arkas.state import ColumnCooccurrenceState
        >>> creator = MatplotlibFigureCreator()
        >>> fig = creator.create(
        ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
        ... )

        ```
        """


class MatplotlibFigureCreator(BaseFigureCreator):
    r"""Create a matplotlib figure of the pairwise column co-occurrence
    matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.figure import MatplotlibFigureConfig
    >>> from arkas.plotter.column_cooccurrence import MatplotlibFigureCreator
    >>> from arkas.state import ColumnCooccurrenceState
    >>> creator = MatplotlibFigureCreator()
    >>> fig = creator.create(
    ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ... )

    ```
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}()"

    def create(self, state: ColumnCooccurrenceState) -> BaseFigure:
        if state.matrix.shape[0] == 0:
            return HtmlFigure(MISSING_FIGURE_MESSAGE)

        fig, ax = plt.subplots(**state.figure_config.get_arg("init", {}))
        im = ax.imshow(state.matrix, norm=state.figure_config.get_arg("color_norm"))
        fig.colorbar(im)
        ax.set_xticks(
            range(len(state.columns)),
            labels=state.columns,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.set_yticks(range(len(state.columns)), labels=state.columns)
        readable_xticklabels(ax, max_num_xticks=50)
        readable_yticklabels(ax, max_num_yticks=50)
        ax.set_title("pairwise column co-occurrence matrix")

        if state.matrix.shape[0] < 16:
            for i in range(len(state.columns)):
                for j in range(len(state.columns)):
                    ax.text(
                        j,
                        i,
                        state.matrix[i, j],
                        ha="center",
                        va="center",
                        color="w",
                        fontsize="xx-small",
                    )

        fig.tight_layout()
        return MatplotlibFigure(fig)


class ColumnCooccurrencePlotter(BaseStateCachedPlotter[ColumnCooccurrenceState]):
    r"""Implement a pairwise column co-occurrence plotter.

    Args:
        state: The state with the co-occurrence matrix.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.plotter import ColumnCooccurrencePlotter
    >>> from arkas.state import ColumnCooccurrenceState
    >>> plotter = ColumnCooccurrencePlotter(
    ...     ColumnCooccurrenceState(matrix=np.ones((3, 3)), columns=["a", "b", "c"])
    ... )
    >>> plotter
    ColumnCooccurrencePlotter(
      (state): ColumnCooccurrenceState(matrix=(3, 3), figure_config=MatplotlibFigureConfig())
    )

    ```
    """

    registry = FigureCreatorRegistry[BaseFigureCreator](
        {MatplotlibFigureConfig.backend(): MatplotlibFigureCreator()}
    )

    def _plot(self) -> dict:
        figure = self.registry.find_creator(self._state.figure_config.backend()).create(self._state)
        return {"column_cooccurrence": figure}
