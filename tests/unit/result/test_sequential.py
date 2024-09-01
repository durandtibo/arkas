from __future__ import annotations

from coola import objects_are_equal

from arkas.result import Result, SequentialResult

######################################
#     Tests for SequentialResult     #
######################################


def test_sequential_result_repr() -> None:
    assert repr(SequentialResult([])).startswith("SequentialResult(")


def test_sequential_result_str() -> None:
    assert str(SequentialResult([])).startswith("SequentialResult(")


def test_sequential_result_equal_true() -> None:
    assert SequentialResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        SequentialResult(
            [
                Result(metrics={"accuracy": 62.0, "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        )
    )


def test_sequential_result_equal_false_different_results() -> None:
    assert not SequentialResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(SequentialResult([Result(metrics={"accuracy": 62.0, "count": 42})]))


def test_sequential_result_equal_false_different_types() -> None:
    assert not SequentialResult(
        [
            Result(metrics={"accuracy": 62.0, "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(Result(metrics={"accuracy": 62.0, "count": 42}))


def test_sequential_result_equal_nan_true() -> None:
    assert SequentialResult(
        [
            Result(metrics={"accuracy": float("nan"), "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        SequentialResult(
            [
                Result(metrics={"accuracy": float("nan"), "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        ),
        equal_nan=True,
    )


def test_sequential_result_equal_nan_false() -> None:
    assert not SequentialResult(
        [
            Result(metrics={"accuracy": float("nan"), "count": 42}),
            Result(metrics={"ap": 0.42, "count": 42}),
        ]
    ).equal(
        SequentialResult(
            [
                Result(metrics={"accuracy": float("nan"), "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        )
    )


def test_sequential_result_compute_metrics() -> None:
    assert objects_are_equal(
        SequentialResult(
            [
                Result(metrics={"accuracy": 62.0, "count": 42}),
                Result(metrics={"ap": 0.42, "count": 42}),
            ]
        ).compute_metrics(),
        {"accuracy": 62.0, "count": 42, "ap": 0.42},
    )


def test_sequential_result_compute_metrics_empty() -> None:
    assert objects_are_equal(SequentialResult([]).compute_metrics(), {})


def test_sequential_result_generate_figures() -> None:
    assert objects_are_equal(
        SequentialResult(
            [
                Result(figures={"accuracy": 62.0, "count": 42}),
                Result(figures={"ap": 0.42, "count": 42}),
            ]
        ).generate_figures(),
        {"accuracy": 62.0, "count": 42, "ap": 0.42},
    )


def test_sequential_result_generate_figures_empty() -> None:
    assert objects_are_equal(SequentialResult([]).generate_figures(), {})
