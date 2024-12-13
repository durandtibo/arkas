from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import JensenShannonDivergenceEvaluator
from arkas.result import EmptyResult, JensenShannonDivergenceResult, Result

######################################################
#     Tests for JensenShannonDivergenceEvaluator     #
######################################################


def test_jensen_shannon_divergence_evaluator_repr() -> None:
    assert repr(JensenShannonDivergenceEvaluator(p="target", q="pred")).startswith(
        "JensenShannonDivergenceEvaluator("
    )


def test_jensen_shannon_divergence_evaluator_str() -> None:
    assert str(JensenShannonDivergenceEvaluator(p="target", q="pred")).startswith(
        "JensenShannonDivergenceEvaluator("
    )


def test_jensen_shannon_divergence_evaluator_evaluate() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="pred")
        .evaluate(pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}))
        .equal(
            JensenShannonDivergenceResult(
                p=np.array([0.4, 0.3, 0.2, 0.1]), q=np.array([0.1, 0.2, 0.3, 0.4])
            )
        )
    )


def test_jensen_shannon_divergence_evaluator_evaluate_lazy_false() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="pred")
        .evaluate(
            pl.DataFrame({"pred": [0.1, 0.6, 0.1, 0.2], "target": [0.1, 0.6, 0.1, 0.2]}), lazy=False
        )
        .equal(Result(metrics={"size": 4, "jensen_shannon_divergence": 0.0}))
    )


def test_jensen_shannon_divergence_evaluator_evaluate_missing_keys() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="missing")
        .evaluate(pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}))
        .equal(EmptyResult())
    )


def test_jensen_shannon_divergence_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="missing")
        .evaluate(
            pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}), lazy=False
        )
        .equal(EmptyResult())
    )


def test_jensen_shannon_divergence_evaluator_evaluate_drop_nulls() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="pred")
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [0.3, 0.2, 0.0, 0.1, 0.0, None, 0.1, None],
                    "target": [0.1, 0.2, 0.3, 0.2, 0.1, 0.2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            JensenShannonDivergenceResult(
                p=np.array([0.1, 0.2, 0.3, 0.2, 0.1]), q=np.array([0.3, 0.2, 0.0, 0.1, 0.0])
            )
        )
    )


def test_jensen_shannon_divergence_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        JensenShannonDivergenceEvaluator(p="target", q="pred", drop_nulls=False)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [0.3, 0.2, 0.0, 0.1, 0.0, None, 0.1, None],
                    "target": [0.1, 0.2, 0.3, 0.2, 0.1, 0.2, None, None],
                    "col": [1, None, 3, 4, 5, None, 7, None],
                }
            )
        )
        .equal(
            JensenShannonDivergenceResult(
                p=np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, float("nan"), float("nan")]),
                q=np.array([0.3, 0.2, 0, 0.1, 0, float("nan"), 0.1, float("nan")]),
            ),
            equal_nan=True,
        )
    )
