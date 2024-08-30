from coola import objects_are_equal

from arkas.run import Run

#########################
#     Tests for Run     #
#########################


def test_run_repr() -> None:
    assert repr(Run(uri="my_uri")).startswith("Run(")


def test_run_str() -> None:
    assert repr(Run(uri="my_uri")).startswith("Run(")


def test_run_get_uri() -> None:
    assert Run(uri="my_uri").get_uri() == "my_uri"


def test_run_get_data() -> None:
    assert objects_are_equal(
        Run(uri="my_uri", data={"accuracy": 0.42, "precision": 0.7}).get_data("accuracy"), 0.42
    )


def test_run_get_metrics() -> None:
    assert objects_are_equal(
        Run(uri="my_uri", metrics={"accuracy": 0.42, "precision": 0.7}).get_metrics(),
        {"accuracy": 0.42, "precision": 0.7},
    )


def test_run_get_metrics_empty() -> None:
    assert objects_are_equal(Run(uri="my_uri").get_metrics(), {})
