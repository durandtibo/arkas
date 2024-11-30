from __future__ import annotations

import numpy as np
import pytest
from coola import objects_are_equal

from arkas.metric.utils import (
    check_array_ndim,
    check_label_type,
    check_nan_policy,
    check_nan_pred,
    check_same_shape,
    check_same_shape_pred,
    check_same_shape_score,
    contains_nan,
    multi_isnan,
    preprocess_pred,
    preprocess_pred_multilabel,
    preprocess_same_shape_arrays,
    preprocess_score_binary,
    preprocess_score_multiclass,
    preprocess_score_multilabel,
)

NAN_POLICIES = ["omit", "propagate", "raise"]


######################################
#     Tests for check_array_ndim     #
######################################


@pytest.mark.parametrize("shape", [(2,), (1,), (3,)])
def test_check_array_ndim_1(shape: tuple[int, ...]) -> None:
    check_array_ndim(arr=np.ones(shape), ndim=1)


@pytest.mark.parametrize("shape", [(2, 3), (1, 1), (3, 2)])
def test_check_array_ndim_2(shape: tuple[int, ...]) -> None:
    check_array_ndim(arr=np.ones(shape), ndim=2)


def test_check_array_ndim_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect number of array dimensions"):
        check_array_ndim(np.ones((2, 3)), ndim=3)


######################################
#     Tests for check_label_type     #
######################################


@pytest.mark.parametrize("label_type", ["binary", "multiclass", "multilabel", "auto"])
def test_check_label_type_valid(label_type: str) -> None:
    check_label_type(label_type)


def test_check_label_type_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'label_type': incorrect"):
        check_label_type("incorrect")


######################################
#     Tests for check_nan_policy     #
######################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_check_nan_policy_valid(nan_policy: str) -> None:
    check_nan_policy(nan_policy)


def test_check_nan_policy_incorrect() -> None:
    with pytest.raises(ValueError, match="Incorrect 'nan_policy': incorrect"):
        check_nan_policy("incorrect")


####################################
#     Tests for check_nan_pred     #
####################################


def test_check_nan_pred_no_nan() -> None:
    check_nan_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1]))


def test_check_nan_pred_y_true_nan() -> None:
    with pytest.raises(RuntimeError, match="'y_true' contains at least one NaN value"):
        check_nan_pred(y_true=np.array([1, 0, 0, 1, np.nan]), y_pred=np.array([0, 1, 0, 1, 1]))


def test_check_nan_pred_y_pred_nan() -> None:
    with pytest.raises(RuntimeError, match="'y_pred' contains at least one NaN value"):
        check_nan_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, np.nan, 1]))


def test_check_nan_pred_both_nan() -> None:
    with pytest.raises(RuntimeError, match="'y_true' contains at least one NaN value"):
        check_nan_pred(y_true=np.array([1, 0, 0, 1, np.nan]), y_pred=np.array([0, 1, np.nan, 1, 1]))


######################################
#     Tests for check_same_shape     #
######################################


def test_check_same_shape_1_array() -> None:
    check_same_shape([np.array([1, 0, 0, 1, 1])])


def test_check_same_shape_2_arrays_correct() -> None:
    check_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5])])


def test_check_same_shape_2_arrays_incorrect() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        check_same_shape([np.array([1, 0, 0, 1, 1]), np.array([1, 0, 0, 1])])


def test_check_same_shape_3_arrays_correct() -> None:
    check_same_shape(
        [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4, 5]), np.array([5, 4, 3, 2, 1])]
    )


def test_check_same_shape_3_arrays_incorrect() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        check_same_shape(
            [np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4]), np.array([6, 5, 4, 3, 2, 1])]
        )


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


##################################
#     Tests for contains_nan     #
##################################


@pytest.mark.parametrize("nan_policy", NAN_POLICIES)
def test_contains_nan_no_nan(nan_policy: str) -> None:
    assert not contains_nan(np.array([1, 2, 3, 4, 5]), nan_policy=nan_policy)


def test_contains_nan_omit() -> None:
    assert contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="omit")


def test_contains_nan_propagate() -> None:
    assert contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="propagate")


def test_contains_nan_raise() -> None:
    with pytest.raises(ValueError, match="input array contains at least one NaN value"):
        contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="raise")


def test_contains_nan_raise_name() -> None:
    with pytest.raises(ValueError, match="'x' contains at least one NaN value"):
        contains_nan(np.array([1, 2, 3, 4, np.nan]), nan_policy="raise", name="'x'")


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


def test_preprocess_pred_drop_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])),
    )


def test_preprocess_pred_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_pred_remove_y_pred_nan() -> None:
    assert objects_are_equal(
        preprocess_pred(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            y_pred=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_pred_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        preprocess_pred(y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1, 0]))


################################################
#     Tests for preprocess_pred_multilabel     #
################################################


def test_preprocess_pred_multilabel_1d() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1])),
        (np.array([[1], [0], [0], [1], [1]]), np.array([[0], [1], [0], [1], [1]])),
    )


def test_preprocess_pred_multilabel_2d() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_pred=np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        ),
        (
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]]),
        ),
    )


def test_preprocess_pred_multilabel_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(
            y_true=np.array(
                [
                    [1.0, float("nan"), 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            y_pred=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, float("nan")],
                ]
            ),
        ),
        (
            np.array(
                [
                    [1.0, float("nan"), 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, float("nan")],
                ]
            ),
        ),
        equal_nan=True,
    )


def test_preprocess_pred_multilabel_drop_nan() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(
            y_true=np.array(
                [
                    [1.0, float("nan"), 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            y_pred=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, float("nan")],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        ),
    )


def test_preprocess_pred_multilabel_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(
            y_true=np.array(
                [
                    [1.0, float("nan"), 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            y_pred=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        ),
    )


def test_preprocess_pred_multilabel_remove_y_pred_nan() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(
            y_true=np.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            y_pred=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, float("nan")],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
        ),
    )


def test_preprocess_pred_multilabel_empty() -> None:
    assert objects_are_equal(
        preprocess_pred_multilabel(y_true=np.array([]), y_pred=np.array([])),
        (np.array([]), np.array([])),
    )


def test_preprocess_pred_multilabel_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_pred' have different shapes"):
        preprocess_pred_multilabel(
            y_true=np.array([1, 0, 0, 1, 1]), y_pred=np.array([0, 1, 0, 1, 1, 0])
        )


def test_preprocess_pred_multilabel_incorrect_ndim_y_true() -> None:
    with pytest.raises(
        RuntimeError, match="'y_true' must be a 1d or 2d array but received an array of shape"
    ):
        preprocess_pred_multilabel(y_true=np.ones((5, 3, 1)), y_pred=np.ones((5, 3)))


##################################################
#     Tests for preprocess_same_shape_arrays     #
##################################################


def test_preprocess_same_shape_arrays_keep_nan_1_array() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays([np.array([1, 0, 0, 1, float("nan")])]),
        (np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),),
        equal_nan=True,
    )


def test_preprocess_same_shape_arrays_keep_nan_2_arrays() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays(
            [np.array([1, 0, 0, 1, float("nan")]), np.array([1, 2, 3, float("nan"), 5])]
        ),
        (
            np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
            np.array([1.0, 2.0, 3.0, float("nan"), 5.0]),
        ),
        equal_nan=True,
    )


def test_preprocess_same_shape_arrays_keep_nan_3_arrays() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays(
            [
                np.array([1, 0, 0, 1, float("nan")]),
                np.array([1, 2, 3, float("nan"), 5]),
                np.array([float("nan"), 4, 3, 2, 1]),
            ]
        ),
        (
            np.array([1.0, 0.0, 0.0, 1.0, float("nan")]),
            np.array([1.0, 2.0, 3.0, float("nan"), 5.0]),
            np.array([float("nan"), 4.0, 3.0, 2.0, 1.0]),
        ),
        equal_nan=True,
    )


def test_preprocess_same_shape_arrays_drop_nan_1_array() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays([np.array([1, 0, 0, 1, float("nan")])], drop_nan=True),
        (np.array([1.0, 0.0, 0.0, 1.0]),),
    )


def test_preprocess_same_shape_arrays_drop_nan_2_arrays() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays(
            [np.array([1, 0, 0, 1, float("nan")]), np.array([1, 2, 3, float("nan"), 5])],
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0])),
    )


def test_preprocess_same_shape_arrays_drop_nan_3_arrays() -> None:
    assert objects_are_equal(
        preprocess_same_shape_arrays(
            [
                np.array([1, 0, 0, 1, float("nan")]),
                np.array([1, 2, 3, float("nan"), 5]),
                np.array([float("nan"), 4, 3, 2, 1]),
            ],
            drop_nan=True,
        ),
        (np.array([0.0, 0.0]), np.array([2.0, 3.0]), np.array([4.0, 3.0])),
    )


def test_preprocess_same_shape_arrays_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="arrays have different shapes"):
        preprocess_same_shape_arrays([np.array([1, 0, 0, 1, 1]), np.array([1, 2, 3, 4])])


#############################################
#     Tests for preprocess_score_binary     #
#############################################


def test_preprocess_score_binary_1d() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1])
        ),
        (np.array([1, 0, 0, 1, 1]), np.array([0, 1, 0, 1, 1])),
    )


def test_preprocess_score_binary_2d() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([[1, 0, 0], [1, 1, 1]]), y_score=np.array([[0, 1, 0], [1, 1, 0]])
        ),
        (np.array([1, 0, 0, 1, 1, 1]), np.array([0, 1, 0, 1, 1, 0])),
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


def test_preprocess_score_binary_drop_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0])),
    )


def test_preprocess_score_binary_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, float("nan")]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 1.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_score_binary_remove_y_score_nan() -> None:
    assert objects_are_equal(
        preprocess_score_binary(
            y_true=np.array([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
            y_score=np.array([0.0, 1.0, 0.0, 1.0, float("nan"), 1.0]),
            drop_nan=True,
        ),
        (np.array([1.0, 0.0, 0.0, 1.0, 0.0]), np.array([0.0, 1.0, 0.0, 1.0, 1.0])),
    )


def test_preprocess_score_binary_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        preprocess_score_binary(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1, 0])
        )


#################################################
#     Tests for preprocess_score_multiclass     #
#################################################


def test_preprocess_score_multiclass_1d() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        (
            np.array([0, 0, 1, 1, 2, 2]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
    )


def test_preprocess_score_multiclass_2d() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([[0], [0], [1], [1], [2], [2]]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        (
            np.array([0, 0, 1, 1, 2, 2]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
    )


def test_preprocess_score_multiclass_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([0, 0, 1, 1, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, float("nan")],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        (
            np.array([0, 0, 1, 1, 2, float("nan")]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, float("nan")],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
        equal_nan=True,
    )


def test_preprocess_score_multiclass_drop_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([0, 0, 1, 1, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, float("nan")],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([0.0, 0.0, 1.0, 2.0]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                ]
            ),
        ),
    )


def test_preprocess_score_multiclass_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([0, 0, 1, 1, 2, float("nan")]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([0.0, 0.0, 1.0, 1.0, 2.0]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                ]
            ),
        ),
    )


def test_preprocess_score_multiclass_remove_y_score_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(
            y_true=np.array([0, 0, 1, 1, 2, 2]),
            y_score=np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.1, 0.8, float("nan")],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
            drop_nan=True,
        ),
        (
            np.array([0, 0, 1, 2, 2]),
            np.array(
                [
                    [0.7, 0.2, 0.1],
                    [0.4, 0.3, 0.3],
                    [0.2, 0.5, 0.3],
                    [0.3, 0.2, 0.5],
                    [0.1, 0.2, 0.7],
                ]
            ),
        ),
    )


def test_preprocess_score_multiclass_empty() -> None:
    assert objects_are_equal(
        preprocess_score_multiclass(y_true=np.array([]), y_score=np.array([])),
        (np.array([]), np.array([])),
    )


def test_preprocess_score_multiclass_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different first dimension"):
        preprocess_score_multiclass(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1, 0])
        )


def test_preprocess_score_multiclass_incorrect_ndim_y_true() -> None:
    with pytest.raises(RuntimeError, match=r"'y_true' must be a an array of shape"):
        preprocess_score_multiclass(y_true=np.ones((5, 3)), y_score=np.ones((5, 3)))


def test_preprocess_score_multiclass_incorrect_ndim_y_score() -> None:
    with pytest.raises(
        RuntimeError, match="'y_score' must be a 2d array but received an array of shape"
    ):
        preprocess_score_multiclass(y_true=np.ones((5,)), y_score=np.ones((5,)))


#################################################
#     Tests for preprocess_score_multilabel     #
#################################################


def test_preprocess_score_multilabel_1d() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1])),
        (np.array([[1], [0], [0], [1], [1]]), np.array([[0], [1], [0], [1], [1]])),
    )


def test_preprocess_score_multilabel_2d() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
        ),
        (
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
        ),
    )


def test_preprocess_score_multilabel_keep_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(
            y_true=np.array(
                [
                    [1.0, float("nan"), 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            y_score=np.array(
                [
                    [2.0, -1.0, -1.0],
                    [-1.0, 1.0, 2.0],
                    [0.0, 2.0, 3.0],
                    [3.0, -2.0, -4.0],
                    [1.0, float("nan"), -5.0],
                ]
            ),
        ),
        (
            np.array([[1, float("nan"), 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, float("nan"), -5]]),
        ),
        equal_nan=True,
    )


def test_preprocess_score_multilabel_drop_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(
            y_true=np.array([[1, float("nan"), 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array(
                [[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, float("nan"), -5]]
            ),
            drop_nan=True,
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]]),
            np.array([[-1.0, 1.0, 2.0], [0.0, 2.0, 3.0], [3.0, -2.0, -4.0]]),
        ),
    )


def test_preprocess_score_multilabel_remove_y_true_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(
            y_true=np.array([[1, float("nan"), 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array([[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
            drop_nan=True,
        ),
        (
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            np.array([[-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, -3, -5]]),
        ),
    )


def test_preprocess_score_multilabel_remove_y_score_nan() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(
            y_true=np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1], [1, 0, 1]]),
            y_score=np.array(
                [[2, -1, -1], [-1, 1, 2], [0, 2, 3], [3, -2, -4], [1, float("nan"), -5]]
            ),
            drop_nan=True,
        ),
        (
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0], [1, 0, 1]]),
            np.array([[2.0, -1.0, -1.0], [-1.0, 1.0, 2.0], [0.0, 2.0, 3.0], [3.0, -2.0, -4.0]]),
        ),
        show_difference=True,
    )


def test_preprocess_score_multilabel_empty() -> None:
    assert objects_are_equal(
        preprocess_score_multilabel(y_true=np.array([]), y_score=np.array([])),
        (np.array([]), np.array([])),
    )


def test_preprocess_score_multilabel_incorrect_shapes() -> None:
    with pytest.raises(RuntimeError, match="'y_true' and 'y_score' have different shapes"):
        preprocess_score_multilabel(
            y_true=np.array([1, 0, 0, 1, 1]), y_score=np.array([0, 1, 0, 1, 1, 0])
        )


def test_preprocess_score_multilabel_incorrect_ndim_y_true() -> None:
    with pytest.raises(
        RuntimeError, match="'y_true' must be a 1d or 2d array but received an array of shape"
    ):
        preprocess_score_multilabel(y_true=np.ones((5, 3, 1)), y_score=np.ones((5, 3)))
