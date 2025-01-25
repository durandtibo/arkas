r"""Implement the pairwise column correlation evaluator."""

from __future__ import annotations

__all__ = ["ColumnCorrelationEvaluator"]

from typing import TYPE_CHECKING, Any

from coola.utils import repr_indent, repr_mapping, str_indent, str_mapping
from grizz.utils.imports import is_tqdm_available

from arkas.evaluator2.caching import BaseCacheEvaluator
from arkas.evaluator2.vanilla import Evaluator
from arkas.metric import pearsonr, spearmanr

if is_tqdm_available():
    from tqdm import tqdm
else:  # pragma: no cover
    from grizz.utils.noop import tqdm

if TYPE_CHECKING:
    from arkas.state.target_dataframe import TargetDataFrameState


class ColumnCorrelationEvaluator(BaseCacheEvaluator):
    r"""Implement the column correlation evaluator.

    Args:
        state: The state with the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator2 import ColumnCorrelationEvaluator
    >>> from arkas.state import TargetDataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    ...         "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...     },
    ... )
    >>> evaluator = ColumnCorrelationEvaluator(
    ...     TargetDataFrameState(frame, target_column="col3")
    ... )
    >>> evaluator
    ColumnCorrelationEvaluator(
      (state): TargetDataFrameState(dataframe=(7, 3), target_column='col3', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> evaluator.evaluate()
    {'col1': {'count': 7, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0, 'spearman_coeff': 1.0, 'spearman_pvalue': 0.0},
     'col2': {'count': 7, 'pearson_coeff': -1.0, 'pearson_pvalue': 0.0, 'spearman_coeff': -1.0, 'spearman_pvalue': 0.0}}

    ```
    """

    def __init__(self, state: TargetDataFrameState) -> None:
        super().__init__()
        self._state = state

    def __repr__(self) -> str:
        args = repr_indent(repr_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def __str__(self) -> str:
        args = str_indent(str_mapping({"state": self._state}))
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def compute(self) -> Evaluator:
        return Evaluator(metrics=self.evaluate())

    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return self._state.equal(other._state, equal_nan=equal_nan)

    def _evaluate(self) -> dict[str, dict[str, float]]:
        target_column = self._state.target_column
        columns = list(self._state.dataframe.columns)
        columns.remove(target_column)

        out = {}
        for col in tqdm(columns, desc="computing correlation"):
            frame = self._state.dataframe.select([col, target_column]).drop_nulls().drop_nans()
            x = frame[target_column].to_numpy()
            y = frame[col].to_numpy()
            out[col] = pearsonr(x, y) | spearmanr(x, y)
        return out
