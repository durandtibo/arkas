from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric.utils import check_nan_true_pred, multi_isnan, preprocess_true_pred

#################################
#     Tests for multi_isnan     #
#################################


def test_multi_isnan_1_array() -> None:
    assert objects_are_equal(
        multi_isnan([np.array([1, 0, 0, 1, float("nan")])]),
        np.array([False, False, False, False, True]),
    )


def test_multi_isnan_2_arrays() -> None:
    assert objects_are_equal(
        multi_isnan(
            [
                np.array([1, 0, 0, 1, float("nan"), float("nan")]),
                np.array([1, float("nan"), 0, 1, 1, float("nan")]),
            ]
        ),
        np.array([False, True, False, False, True, True]),
    )


def test_multi_isnan_3_arrays() -> None:
    assert objects_are_equal(
        multi_isnan(
            [
                np.array([1, 0, 0, 1, float("nan"), float("nan")]),
                np.array([1, float("nan"), 0, 1, 1, float("nan")]),
                np.array([float("nan"), 1, 0, 1, 1, float("nan")]),
            ]
        ),
        np.array([True, True, False, False, True, True]),
    )


def test_multi_isnan_empty() -> None:
    with pytest.raises(RuntimeError, match="'arrays' cannot be empty"):
        multi_isnan([])


##########################################
#     Tests for preprocess_true_pred     #
##########################################


def test_preprocess_true_pred_no_nan() -> None:
    assert objects_are_equal(
        preprocess_true_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1])),
        (np.array([1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 1])),
    )


def test_preprocess_true_pred_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_true_pred(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        (
            np.array([1, 0, 0, 1, 1, float("nan")]),
            np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_preprocess_true_pred_remove_nan() -> None:
    assert objects_are_equal(
        preprocess_true_pred(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([0, 1, 0, 1, float("nan"), 1]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])),
    )


def test_preprocess_true_pred_nan_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'nan': incorrect"):
        preprocess_true_pred(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1]), nan="incorrect"
        )


#########################################
#     Tests for check_nan_true_pred     #
#########################################


def test_check_nan_true_pred_valid() -> None:
    check_nan_true_pred("keep")
    check_nan_true_pred("remove")


def test_check_nan_true_pred_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'nan': incorrect"):
        check_nan_true_pred(nan="incorrect")
