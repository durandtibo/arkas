from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric.utils import (
    check_label_type,
    check_nan_option,
    check_same_shape_pred,
    check_same_shape_score,
    multi_isnan,
    preprocess_pred,
    preprocess_score_binary,
)

######################################
#     Tests for check_label_type     #
######################################


@pytest.mark.parametrize("label_type", ["binary", "multiclass", "multilabel", "auto"])
def test_check_label_type_valid(label_type: str) -> None:
    check_label_type(label_type)


def test_check_label_type_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'label_type': incorrect"):
        check_label_type("incorrect")


######################################
#     Tests for check_nan_option     #
######################################


def test_check_nan_option_valid() -> None:
    check_nan_option("keep")
    check_nan_option("remove")


def test_check_nan_option_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'nan': incorrect"):
        check_nan_option(nan="incorrect")


###########################################
#     Tests for check_same_shape_pred     #
###########################################


def test_check_same_shape_pred_1d() -> None:
    check_same_shape_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1]))


def test_check_same_shape_pred_2d() -> None:
    check_same_shape_pred(
        y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_pred=np.array([[1, 0, 0], [1, 1, 1]])
    )


def test_check_same_shape_pred_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        check_same_shape_pred(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_pred=np.array([0, 1, 0, 1, 1, 0]),
        )


############################################
#     Tests for check_same_shape_score     #
############################################


def test_check_same_shape_score_1d() -> None:
    check_same_shape_score(y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1]))


def test_check_same_shape_score_2d() -> None:
    check_same_shape_score(
        y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_score=np.array([[1, 0, 0], [1, 1, 1]])
    )


def test_check_same_shape_score_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        check_same_shape_score(
            y_true=np.array([1, 0, 0, 1, 1]),
            y_score=np.array([0, 1, 0, 1, 1, 0]),
        )


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


#####################################
#     Tests for preprocess_pred     #
#####################################


def test_preprocess_pred_no_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1])),
        (np.array([1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 1])),
    )


def test_preprocess_pred_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_pred=np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        (
            np.array([1, 0, 0, 1, 1, float("nan")]),
            np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_preprocess_pred_remove_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])),
    )


def test_preprocess_pred_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_pred_remove_y_pred_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_pred_nan_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'nan': incorrect"):
        preprocess_pred(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1]), nan="incorrect"
        )


def test_preprocess_pred_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        preprocess_pred(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1, 0]), nan="keep"
        )


#############################################
#     Tests for preprocess_score_binary     #
#############################################


def test_preprocess_score_binary_no_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1])
        ),
        (np.array([1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 1])),
    )


def test_preprocess_score_binary_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1, float("nan")]),
            y_score=np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        (
            np.array([1, 0, 0, 1, 1, float("nan")]),
            np.array([0, 1, 0, 1, float("nan"), 1]),
        ),
        equal_nan=True,
    )


def test_preprocess_score_binary_remove_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])),
    )


def test_preprocess_score_binary_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_score_binary_remove_y_score_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            nan="remove",
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_score_binary_nan_incorrect() -> None:
    with pytest.raises(RuntimeError, match="Incorrect 'nan': incorrect"):
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1]), nan="incorrect"
        )


def test_preprocess_score_binary_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1, 0]), nan="keep"
        )
