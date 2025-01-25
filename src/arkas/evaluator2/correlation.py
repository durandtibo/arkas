r"""Implement the pairwise column correlation evaluator."""

from __future__ import annotations

__all__ = ["CorrelationEvaluator"]


from arkas.evaluator2.caching import BaseStateCachedEvaluator
from arkas.metric import pearsonr, spearmanr
from arkas.state.target_dataframe import DataFrameState
from arkas.utils.dataframe import check_num_columns


class CorrelationEvaluator(BaseStateCachedEvaluator[DataFrameState]):
    r"""Implement the pairwise column correlation evaluator.

    Args:
        state: The state with the DataFrame to analyze.
            The DataFrame must have only 2 columns, which are the two
            columns to analyze.

    Example usage:

    ```pycon

    >>> import polars as pl
    >>> from arkas.evaluator2 import CorrelationEvaluator
    >>> from arkas.state import DataFrameState
    >>> frame = pl.DataFrame(
    ...     {
    ...         "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    ...         "col3": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    ...     },
    ... )
    >>> evaluator = CorrelationEvaluator(DataFrameState(frame))
    >>> evaluator
    CorrelationEvaluator(
      (state): DataFrameState(dataframe=(7, 2), nan_policy='propagate', figure_config=MatplotlibFigureConfig())
    )
    >>> evaluator.evaluate()
    {'count': 7, 'pearson_coeff': 1.0, 'pearson_pvalue': 0.0, 'spearman_coeff': 1.0, 'spearman_pvalue': 0.0}

    ```
    """

    def __init__(self, state: DataFrameState) -> None:
        check_num_columns(state.dataframe, num_columns=2)
        super().__init__(state=state)

    def _evaluate(self) -> dict[str, float]:
        frame = self._state.dataframe
        x = frame[frame.columns[0]].to_numpy()
        y = frame[frame.columns[1]].to_numpy()
        return pearsonr(x=x, y=y, nan_policy=self._state.nan_policy) | spearmanr(
            x=x, y=y, nan_policy=self._state.nan_policy
        )
