from __future__ import annotations

import numpy as np
import polars as pl
from grizz.transformer import DropNullRow

from arkas.analyzer import AccuracyAnalyzer, TransformAnalyzer
from arkas.output import AccuracyOutput
from arkas.state import AccuracyState

#######################################
#     Tests for TransformAnalyzer     #
#######################################


def test_transform_analyzer_repr() -> None:
    assert repr(
        TransformAnalyzer(
            transformer=DropNullRow(), analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred")
        )
    ).startswith("AccuracyAnalyzer(")


def test_transform_analyzer_str() -> None:
    assert str(
        TransformAnalyzer(
            transformer=DropNullRow(), analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred")
        )
    ).startswith("AccuracyAnalyzer(")


def test_transform_analyzer_analyze() -> None:
    assert (
        TransformAnalyzer(
            transformer=DropNullRow(), analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred")
        )
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0, None], "target": [1, 2, 3, 2, 1, None]}))
        .equal(
            AccuracyOutput(
                state=AccuracyState(
                    y_true=np.array([1, 2, 3, 2, 1]),
                    y_pred=np.array([3, 2, 0, 1, 0]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            )
        )
    )
