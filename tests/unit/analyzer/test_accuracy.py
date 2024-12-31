from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from grizz.exceptions import ColumnNotFoundError, ColumnNotFoundWarning

from arkas.analyzer import AccuracyAnalyzer
from arkas.output import AccuracyOutput, EmptyOutput
from arkas.state import AccuracyState

######################################
#     Tests for AccuracyAnalyzer     #
######################################


def test_accuracy_analyzer_repr() -> None:
    assert repr(AccuracyAnalyzer(y_true="target", y_pred="pred")).startswith("AccuracyAnalyzer(")


def test_accuracy_analyzer_str() -> None:
    assert str(AccuracyAnalyzer(y_true="target", y_pred="pred")).startswith("AccuracyAnalyzer(")


def test_accuracy_analyzer_analyze() -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred")
        .analyze(pl.DataFrame({"pred": [3, 2, 0, 1, 0], "target": [1, 2, 3, 2, 1]}))
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


def test_accuracy_analyzer_analyze_drop_nulls() -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred")
        .analyze(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
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


def test_accuracy_analyzer_analyze_drop_nulls_false() -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred", drop_nulls=False)
        .analyze(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            AccuracyOutput(
                state=AccuracyState(
                    y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                    y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_accuracy_analyzer_analyze_nan_policy(nan_policy: str) -> None:
    assert (
        AccuracyAnalyzer(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .analyze(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None],
                    "target": [1, 2, 3, 2, 1, None],
                }
            )
        )
        .equal(
            AccuracyOutput(
                state=AccuracyState(
                    y_true=np.array([1, 2, 3, 2, 1]),
                    y_pred=np.array([3, 2, 0, 1, 0]),
                    y_true_name="target",
                    y_pred_name="pred",
                ),
                nan_policy=nan_policy,
            )
        )
    )


def test_accuracy_analyzer_analyze_missing_policy_ignore() -> None:
    frame = pl.DataFrame({"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_ignore_y_true() -> None:
    frame = pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="gt_target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_ignore_y_pred() -> None:
    frame = pl.DataFrame({"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="ignore")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_policy_raise() -> None:
    frame = pl.DataFrame({"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(ColumnNotFoundError, match="column 'target' is missing in the DataFrame"):
        analyzer.analyze(frame)


def test_accuracy_analyzer_analyze_missing_raise_y_true() -> None:
    frame = pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "col": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(ColumnNotFoundError, match="column 'target' is missing in the DataFrame"):
        analyzer.analyze(frame)


def test_accuracy_analyzer_analyze_missing_raise_y_pred() -> None:
    frame = pl.DataFrame({"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred")
    with pytest.raises(ColumnNotFoundError, match="column 'pred' is missing in the DataFrame"):
        analyzer.analyze(frame)


def test_accuracy_analyzer_analyze_missing_policy_warn() -> None:
    frame = pl.DataFrame({"col1": np.array([3, 2, 0, 1, 0]), "col2": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with (
        pytest.warns(
            ColumnNotFoundWarning,
            match="column 'target' is missing in the DataFrame and will be ignored",
        ),
        pytest.warns(
            ColumnNotFoundWarning,
            match="column 'pred' is missing in the DataFrame and will be ignored",
        ),
    ):
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_warn_y_true() -> None:
    frame = pl.DataFrame({"pred": np.array([3, 2, 0, 1, 0]), "col": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning,
        match="column 'target' is missing in the DataFrame and will be ignored",
    ):
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())


def test_accuracy_analyzer_analyze_missing_warn_y_pred() -> None:
    frame = pl.DataFrame({"col": np.array([3, 2, 0, 1, 0]), "target": np.array([1, 2, 3, 2, 1])})
    analyzer = AccuracyAnalyzer(y_true="target", y_pred="pred", missing_policy="warn")
    with pytest.warns(
        ColumnNotFoundWarning, match="column 'pred' is missing in the DataFrame and will be ignored"
    ):
        out = analyzer.analyze(frame)
    assert out.equal(EmptyOutput())
