from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from arkas.evaluator import EnergyDistanceEvaluator
from arkas.result import EmptyResult, EnergyDistanceResult, Result

#############################################
#     Tests for EnergyDistanceEvaluator     #
#############################################


def test_energy_distance_evaluator_repr() -> None:
    assert repr(EnergyDistanceEvaluator(u_values="target", v_values="pred")).startswith(
        "EnergyDistanceEvaluator("
    )


def test_energy_distance_evaluator_str() -> None:
    assert str(EnergyDistanceEvaluator(u_values="target", v_values="pred")).startswith(
        "EnergyDistanceEvaluator("
    )


def test_energy_distance_evaluator_evaluate() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [5, 4, 3, 2, 1]}))
        .equal(
            EnergyDistanceResult(
                u_values=np.array([5, 4, 3, 2, 1]), v_values=np.array([1, 2, 3, 4, 5])
            )
        )
    )


def test_energy_distance_evaluator_evaluate_lazy_false() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="pred")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(Result(metrics={"count": 5, "energy_distance": 0.0}))
    )


def test_energy_distance_evaluator_evaluate_missing_keys() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}))
        .equal(EmptyResult())
    )


def test_energy_distance_evaluator_evaluate_lazy_false_missing_keys() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="missing")
        .evaluate(pl.DataFrame({"pred": [1, 2, 3, 4, 5], "target": [1, 2, 3, 4, 5]}), lazy=False)
        .equal(EmptyResult())
    )


def test_energy_distance_evaluator_evaluate_drop_nulls() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="pred")
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
            EnergyDistanceResult(
                u_values=np.array([1, 2, 3, 2, 1]), v_values=np.array([3, 2, 0, 1, 0])
            )
        )
    )


def test_energy_distance_evaluator_evaluate_drop_nulls_false() -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="pred", drop_nulls=False)
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
            EnergyDistanceResult(
                u_values=np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, float("nan"), float("nan")]),
                v_values=np.array([3.0, 2.0, 0.0, 1.0, 0.0, float("nan"), 1.0, float("nan")]),
            ),
            equal_nan=True,
        )
    )


@pytest.mark.parametrize("nan_policy", ["omit", "propagate", "raise"])
def test_energy_distance_evaluator_evaluate_nan_policy(nan_policy: str) -> None:
    assert (
        EnergyDistanceEvaluator(u_values="target", v_values="pred", nan_policy=nan_policy)
        .evaluate(
            pl.DataFrame(
                {
                    "pred": [1.0, 2.0, 3.0, 4.0, 5.0, float("nan")],
                    "target": [5.0, 4.0, 3.0, 2.0, 1.0, float("nan")],
                }
            )
        )
        .equal(
            EnergyDistanceResult(
                u_values=np.array([5.0, 4.0, 3.0, 2.0, 1.0, float("nan")]),
                v_values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, float("nan")]),
                nan_policy=nan_policy,
            ),
            equal_nan=True,
        )
    )
