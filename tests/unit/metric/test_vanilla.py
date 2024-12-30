from __future__ import annotations

from coola import objects_are_equal

from arkas.metric import EmptyMetric, Metric

############################
#     Tests for Metric     #
############################


def test_metric_repr() -> None:
    assert repr(Metric()) == "Metric(count=0)"


def test_metric_str() -> None:
    assert str(Metric()) == "Metric(count=0)"


def test_metric_equal_true() -> None:
    assert Metric(metrics={"accuracy": 1.0, "count": 42}).equal(
        Metric(metrics={"accuracy": 1.0, "count": 42})
    )


def test_metric_equal_false_different_metrics() -> None:
    assert not Metric(metrics={"accuracy": 1.0, "count": 42}).equal(
        Metric(metrics={"accuracy": 1.0})
    )


def test_metric_equal_false_different_type() -> None:
    assert not Metric().equal(42)


def test_metric_equal_nan_true() -> None:
    assert Metric(metrics={"accuracy": float("nan"), "count": 42}).equal(
        Metric(metrics={"accuracy": float("nan"), "count": 42}),
        equal_nan=True,
    )


def test_metric_equal_nan_false() -> None:
    assert not Metric(metrics={"accuracy": float("nan"), "count": 42}).equal(
        Metric(metrics={"accuracy": float("nan"), "count": 42})
    )


def test_metric_evaluate() -> None:
    assert objects_are_equal(
        Metric(metrics={"accuracy": 1.0, "count": 42}).evaluate(),
        {"accuracy": 1.0, "count": 42},
    )


def test_metric_evaluate_empty() -> None:
    assert objects_are_equal(Metric().evaluate(), {})


def test_metric_evaluate_prefix_suffix() -> None:
    assert objects_are_equal(
        Metric(metrics={"accuracy": 1.0, "count": 42}).evaluate(prefix="prefix_", suffix="_suffix"),
        {"prefix_accuracy_suffix": 1.0, "prefix_count_suffix": 42},
    )


#################################
#     Tests for EmptyMetric     #
#################################


def test_empty_metric_repr() -> None:
    assert repr(EmptyMetric()) == "EmptyMetric()"


def test_empty_metric_str() -> None:
    assert str(EmptyMetric()) == "EmptyMetric()"


def test_empty_metric_evaluate() -> None:
    assert objects_are_equal(EmptyMetric().evaluate(), {})
