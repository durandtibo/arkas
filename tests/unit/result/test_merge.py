from __future__ import annotations

from coola import objects_are_equal

from arkas.result import MergedResult, Result

##################################
#     Tests for MergedResult     #
##################################


def test_merged_result_repr() -> None:
    assert repr(MergedResult([])).startswith("MergedResult(")


def test_merged_result_str() -> None:
    assert str(MergedResult([])).startswith("MergedResult(")


def test_merged_result_equal_true() -> None:
    assert MergedResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        MergedResult(
            [
                Result(metrics={"accuracy": 62.0, "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        )
    )


def test_merged_result_equal_false_different_results() -> None:
    assert not MergedResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(MergedResult([Result(metrics={"accuracy": 62.0, "count": 42})]))


def test_merged_result_equal_false_different_types() -> None:
    assert not MergedResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(Result(metrics={"accuracy": 62.0, "count": 42}))


def test_merged_result_equal_nan_true() -> None:
    assert MergedResult(
        [
            Result(metrics={"accuracy": float("nan"), "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        MergedResult(
            [
                Result(metrics={"accuracy": float("nan"), "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        ),
        equal_nan=True,
    )


def test_merged_result_equal_nan_false() -> None:
    assert not MergedResult(
        [
            Result(metrics={"accuracy": float("nan"), "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        MergedResult(
            [
                Result(metrics={"accuracy": float("nan"), "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        )
    )


def test_merged_result_compute_metrics() -> None:
    assert objects_are_equal(
        MergedResult(
            [
                Result(metrics={"accuracy": 62.0, "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        ).compute_metrics(),
        {"accuracy": 62.0, "count": 42, "ap": 0.42},
    )


def test_merged_result_compute_metrics_empty() -> None:
    assert objects_are_equal(MergedResult([]).compute_metrics(), {})


def test_merged_result_generate_figures() -> None:
    assert objects_are_equal(
        MergedResult(
            [
                Result(figures={"accuracy": 62.0, "count": 42}),
                Result(figures={"ap": 0.42, "count": 42}),
            ]
        ).generate_figures(),
        {"accuracy": 62.0, "count": 42, "ap": 0.42},
    )


def test_merged_result_generate_figures_empty() -> None:
    assert objects_are_equal(MergedResult([]).generate_figures(), {})
