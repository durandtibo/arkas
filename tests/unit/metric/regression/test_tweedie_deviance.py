from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric import mean_tweedie_deviance

###########################################
#     Tests for mean_tweedie_deviance     #
###########################################


def test_mean_tweedie_deviance_correct() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5])),
        {"count": 5, "mean_tweedie_deviance_power_0": 0.0},
    )


def test_mean_tweedie_deviance_correct_2d() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([[1, 2, 3], [4, 5, 6]]), y_pred=np.array([[1, 2, 3], [4, 5, 6]])
        ),
        {"count": 6, "mean_tweedie_deviance_power_0": 0.0},
    )


def test_mean_tweedie_deviance_incorrect() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(y_true=np.array([4, 3, 2, 1]), y_pred=np.array([1, 2, 3, 4])),
        {"count": 4, "mean_tweedie_deviance_power_0": 5.0},
    )


def test_mean_tweedie_deviance_powers() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5]), y_pred=np.array([1, 2, 3, 4, 5]), powers=[0, 1, 2]
        ),
        {
            "count": 5,
            "mean_tweedie_deviance_power_0": 0.0,
            "mean_tweedie_deviance_power_1": 0.0,
            "mean_tweedie_deviance_power_2": 0.0,
        },
    )


def test_mean_tweedie_deviance_empty() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(y_true=np.array([]), y_pred=np.array([])),
        {"count": 0, "mean_tweedie_deviance_power_0": float("nan")},
        equal_nan=True,
    )


def test_mean_tweedie_deviance_prefix_suffix() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5]),
            y_pred=np.array([1, 2, 3, 4, 5]),
            prefix="prefix_",
            suffix="_suffix",
        ),
        {"prefix_count_suffix": 5, "prefix_mean_tweedie_deviance_power_0_suffix": 0.0},
    )


def test_mean_tweedie_deviance_nan_omit() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="omit",
        ),
        {"count": 3, "mean_tweedie_deviance_power_0": 0.0},
    )


def test_mean_tweedie_deviance_nan_omit_y_true() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="omit",
        ),
        {"count": 5, "mean_tweedie_deviance_power_0": 0.0},
    )


def test_mean_tweedie_deviance_nan_omit_y_pred() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="omit",
        ),
        {"count": 5, "mean_tweedie_deviance_power_0": 0.0},
    )


def test_mean_tweedie_deviance_nan_propagate() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_tweedie_deviance_power_0": float("nan")},
        equal_nan=True,
    )


def test_mean_tweedie_deviance_nan_propagate_y_true() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_tweedie_deviance_power_0": float("nan")},
        equal_nan=True,
    )


def test_mean_tweedie_deviance_nan_propagate_y_pred() -> None:
    assert objects_are_equal(
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="propagate",
        ),
        {"count": 6, "mean_tweedie_deviance_power_0": float("nan")},
        equal_nan=True,
    )


def test_mean_tweedie_deviance_nan_raise() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        mean_tweedie_deviance(
            y_true=np.array([float("nan"), 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, float("nan"), float("nan")]),
            nan_policy="raise",
        )


def test_mean_tweedie_deviance_nan_raise_y_true() -> None:
    with pytest.raises(ValueError, match="'y_true' contains at least one NaN value"):
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, float("nan")]),
            y_pred=np.array([1, 2, 3, 4, 5, 0]),
            nan_policy="raise",
        )


def test_mean_tweedie_deviance_nan_raise_y_pred() -> None:
    with pytest.raises(ValueError, match="'y_pred' contains at least one NaN value"):
        mean_tweedie_deviance(
            y_true=np.array([1, 2, 3, 4, 5, 0]),
            y_pred=np.array([1, 2, 3, 4, 5, float("nan")]),
            nan_policy="raise",
        )
