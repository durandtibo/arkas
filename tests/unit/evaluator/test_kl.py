from __future__ import annotations

import numpy as np
import polars as pl

from arkas.evaluator import KLDivEvaluator
from arkas.result import EmptyResult, KLDivResult, Result

####################################
#     Tests for KLDivEvaluator     #
####################################


def test_kl_div_evaluator_repr() -> None:
    assert repr(KLDivEvaluator(p="target", q="pred")).startswith("KLDivEvaluator(")


def test_kl_div_evaluator_str() -> None:
    assert str(KLDivEvaluator(p="target", q="pred")).startswith("KLDivEvaluator(")


def test_kl_div_evaluator_evaluate() -> None:
    assert (
        KLDivEvaluator(p="target", q="pred")
        .evaluate(pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}))
        .equal(KLDivResult(p=np.array([0.4, 0.3, 0.2, 0.1]), q=np.array([0.1, 0.2, 0.3, 0.4])))
    )


def test_kl_div_evaluator_evaluate_lazy_false() -> None:
    assert (
        KLDivEvaluator(p="target", q="pred")
        .evaluate(
            pl.DataFrame({"pred": [0.1, 0.6, 0.1, 0.2], "target": [0.1, 0.6, 0.1, 0.2]}), lazy=False
        )
        .equal(Result(metrics={"size": 4, "kl_pq": 0.0, "kl_qp": 0.0}))
    )


def test_kl_div_evaluator_evaluate_missing_keys() -> None:
    assert (
        KLDivEvaluator(p="target", q="missing")
        .evaluate(pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}))
        .equal(EmptyResult())
    )


def test_kl_div_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        KLDivEvaluator(p="target", q="missing")
        .evaluate(
            pl.DataFrame({"pred": [0.1, 0.2, 0.3, 0.4], "target": [0.4, 0.3, 0.2, 0.1]}), lazy=False
        )
        .equal(EmptyResult())
    )


def test_kl_div_evaluator_evaluate_drop_nulls() -> None:
    assert (
        KLDivEvaluator(p="target", q="pred")
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
            KLDivResult(
                p=np.array([0.1, 0.2, 0.3, 0.2, 0.1]), q=np.array([0.3, 0.2, 0.0, 0.1, 0.0])
            )
        )
    )


def test_kl_div_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        KLDivEvaluator(p="target", q="pred", drop_nulls=False)
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
            KLDivResult(
                p=np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.2, float("nan"), float("nan")]),
                q=np.array([0.3, 0.2, 0, 0.1, 0, float("nan"), 0.1, float("nan")]),
            ),
            equal_nan=True,
        )
    )
