from __future__ import annotations

from coola import objects_are_equal

from arkas.result import MappingResult, Result

###################################
#     Tests for MappingResult     #
###################################


def test_mapping_result_repr() -> None:
    assert repr(MappingResult({})).startswith("MappingResult(")


def test_mapping_result_str() -> None:
    assert str(MappingResult({})).startswith("MappingResult(")


def test_mapping_result_equal_true() -> None:
    assert MappingResult(
        {
            "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
            "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingResult(
            {
                "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
                "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
            }
        )
    )


def test_mapping_result_equal_false_different_results() -> None:
    assert not MappingResult(
        {
            "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
            "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingResult(
            {
                "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
            }
        )
    )


def test_mapping_result_equal_false_different_types() -> None:
    assert not MappingResult(
        {
            "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
            "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(Result(metrics={"accuracy": 62.0, "count": 42}))


def test_mapping_result_equal_nan_true() -> None:
    assert MappingResult(
        {
            "class1": Result(metrics={"accuracy": float("nan"), "count": 42}),
            "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingResult(
            {
                "class1": Result(metrics={"accuracy": float("nan"), "count": 42}),
                "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
        equal_nan=True,
    )


def test_mapping_result_equal_nan_false() -> None:
    assert not MappingResult(
        {
            "class1": Result(metrics={"accuracy": float("nan"), "count": 42}),
            "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
        }
    ).equal(
        MappingResult(
            {
                "class1": Result(metrics={"accuracy": float("nan"), "count": 42}),
                "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
            }
        ),
    )


def test_mapping_result_compute_metrics() -> None:
    assert objects_are_equal(
        MappingResult(
            {
                "class1": Result(metrics={"accuracy": 62.0, "count": 42}),
                "class2": Result(metrics={"accuracy": 42.0, "count": 30}),
            }
        ).compute_metrics(),
        {"class1": {"accuracy": 62.0, "count": 42}, "class2": {"accuracy": 42.0, "count": 30}},
    )


def test_mapping_result_compute_metrics_empty() -> None:
    assert objects_are_equal(MappingResult({}).compute_metrics(), {})


def test_mapping_result_generate_figures() -> None:
    assert objects_are_equal(
        MappingResult(
            {
                "class1": Result(figures={"accuracy": 62.0}),
                "class2": Result(figures={"accuracy": 42.0}),
            }
        ).generate_figures(),
        {"class1": {"accuracy": 62.0}, "class2": {"accuracy": 42.0}},
    )


def test_mapping_result_generate_figures_empty() -> None:
    assert objects_are_equal(MappingResult({}).generate_figures(), {})
