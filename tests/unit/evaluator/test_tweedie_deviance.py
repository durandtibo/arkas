from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import MeanTweedieDevianceEvaluator
from arkas.result import EmptyResult, MeanTweedieDevianceResult, Result

##################################################
#     Tests for MeanTweedieDevianceEvaluator     #
##################################################


def test_mean_tweedie_deviance_evaluator_repr() -> None:
    assert repr(MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanTweedieDevianceEvaluator("
    )


def test_mean_tweedie_deviance_evaluator_str() -> None:
    assert str(MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred")).startswith(
        "MeanTweedieDevianceEvaluator("
    )


def test_mean_tweedie_deviance_evaluator_evaluate() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [5, 4, 3, 2, 1]}))
        .equal(
            MeanTweedieDevianceResult(
                y_true=np.array([5, 4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4, 5])
            )
        )
    )


def test_mean_tweedie_deviance_evaluator_evaluate_powers() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred", powers=[0, 1, 2])
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [5, 4, 3, 2, 1]}))
        .equal(
            MeanTweedieDevianceResult(
                y_true=np.array([5, 4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4, 5]), powers=[0, 1, 2]
            )
        )
    )


def test_mean_tweedie_deviance_evaluator_evaluate_lazy_false() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(Result(metrics={"count": 5, "mean_tweedie_deviance_power_0": 0.0}))
    )


def test_mean_tweedie_deviance_evaluator_evaluate_missing_keys() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}))
        .equal(EmptyResult())
    )


def test_mean_tweedie_deviance_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(EmptyResult())
    )


def test_mean_tweedie_deviance_evaluator_evaluate_drop_nulls() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            MeanTweedieDevianceResult(
                y_true=np.array([1, 2, 3, 2, 1]), y_pred=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_mean_tweedie_deviance_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [3, 2, 0, 1, 0, None, 1, None],
                    "target": [1, 2, 3, 2, 1, 2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            MeanTweedieDevianceResult(
                y_true=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                y_pred=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_mean_tweedie_deviance_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        MeanTweedieDevianceEvaluator(y_true="target", y_pred="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, None],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, None],
                }
            )
        )
        .equal(
            MeanTweedieDevianceResult(
                y_true=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
                y_pred=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                nan_policy=nan_policy,
            ),
        )
    )
