r"""Implement the pairwise column correlation evaluator."""

from __future__ import annotations

__all__ = ["CorrelationEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.metric import pearsonr, spearmanr
from arkas.state.columns import TwoColumnDataFrameState


class CorrelationEvaluator(BaseStateCachedEvaluator[TwoColumnDataFrameState]):
    r"""Implement the pairwise column correlation evaluator.

    Args:
        state: The state with the DataFrame to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator2 import CorrelationEvaluator
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col2": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     },
    ... )
    >>> evaluator = CorrelationEvaluator(
    ...     TwoColumnDataFrameState(frame, column1="col1", column2="col2")
    ... )
    >>> evaluator
    CorrelationEvaluator(
      (state): TwoColumnDataFrameState(dataframe=(7, 2), column1='col1', column2='col2', nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> evaluator.evaluate()
    {'count': 7, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0, 'spearman_coeff': 1.0, 'spearman_pvalue': 0.0}

    ```
    """

    def __init__(self, state: TwoColumnDataFrameState) -> None:
        super().__init__(state=state)

    def _evaluate(self) -> dict[str, float]:
        x = self._state.dataframe[self._state.column1].to_numpy()
        y = self._state.dataframe[self._state.column2].to_numpy()
        return pearsonr(x=x, y=y, nan_policy=self._state.nan_policy) | spearmanr(
            x=x, y=y, nan_policy=self._state.nan_policy
        )
