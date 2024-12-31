from __future__ import annotations

import numpy as np

from arkas.evaluator2 import AccuracyEvaluator, Evaluator, EvaluatorDict
from arkas.hcg import AccuracyContentGenerator, ContentGenerator, ContentGeneratorDict
from arkas.output import AccuracyOutput, Output, OutputDict
from arkas.plotter import Plotter, PlotterDict
from arkas.state import AccuracyState

################################
#     Tests for OutputDict     #
################################


def test_output_dict_repr() -> None:
    assert repr(OutputDict({})).startswith("OutputDict(")


def test_output_dict_str() -> None:
    assert str(OutputDict({})).startswith("OutputDict(")


def test_output_dict_equal_true() -> None:
    assert OutputDict(
        {
            "one": Output(
                content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
            ),
            "two": AccuracyOutput(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 0, 1, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            ),
        }
    ).equal(
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
    )


def test_output_dict_equal_false_different_outputs() -> None:
    assert not OutputDict(
        {
            "one": Output(
                content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
            ),
            "two": AccuracyOutput(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 0, 1, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            ),
        }
    ).equal(
        OutputDict(
            {"one": Output(content=ContentGenerator(), evaluator=Evaluator(), plotter=Plotter())}
        )
    )


def test_output_dict_equal_false_different_types() -> None:
    assert not OutputDict(
        {
            "one": Output(
                content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
            ),
            "two": AccuracyOutput(
                AccuracyState(
                    y_true=np.array([1, 0, 0, 1, 1]),
                    y_pred=np.array([1, 0, 0, 1, 1]),
                    y_true_name="target",
                    y_pred_name="pred",
                )
            ),
        }
    ).equal(42.0)


def test_output_dict_equal_nan_true() -> None:
    assert OutputDict(
        {
            "one": Output(
                content=ContentGenerator("meow"),
                evaluator=Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                plotter=Plotter(),
            ),
            "two": Output(
                content=ContentGenerator(),
                evaluator=Evaluator(metrics={"f1": float("nan")}),
                plotter=Plotter(),
            ),
        }
    ).equal(
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"),
                    evaluator=Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                    plotter=Plotter(),
                ),
                "two": Output(
                    content=ContentGenerator(),
                    evaluator=Evaluator(metrics={"f1": float("nan")}),
                    plotter=Plotter(),
                ),
            }
        ),
        equal_nan=True,
    )


def test_output_dict_equal_nan_false() -> None:
    assert not OutputDict(
        {
            "one": Output(
                content=ContentGenerator("meow"),
                evaluator=Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                plotter=Plotter(),
            ),
            "two": Output(
                content=ContentGenerator(),
                evaluator=Evaluator(metrics={"f1": float("nan")}),
                plotter=Plotter(),
            ),
        }
    ).equal(
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"),
                    evaluator=Evaluator(metrics={"accuracy": float("nan"), "count": 42}),
                    plotter=Plotter(),
                ),
                "two": Output(
                    content=ContentGenerator(),
                    evaluator=Evaluator(metrics={"f1": float("nan")}),
                    plotter=Plotter(),
                ),
            }
        ),
    )


def test_output_dict_get_content_generator() -> None:
    assert (
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
        .get_content_generator()
        .equal(
            ContentGeneratorDict(
                {
                    "one": ContentGenerator("meow"),
                    "two": AccuracyContentGenerator(
                        AccuracyState(
                            y_true=np.array([1, 0, 0, 1, 1]),
                            y_pred=np.array([1, 0, 0, 1, 1]),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                }
            )
        )
    )


def test_output_dict_plot_get_content_generator_empty() -> None:
    assert OutputDict({}).get_content_generator().equal(ContentGeneratorDict({}))


def test_output_dict_plot_get_evaluator_lazy_true() -> None:
    assert (
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
        .get_evaluator()
        .equal(
            EvaluatorDict(
                {
                    "one": Evaluator(),
                    "two": AccuracyEvaluator(
                        AccuracyState(
                            y_true=np.array([1, 0, 0, 1, 1]),
                            y_pred=np.array([1, 0, 0, 1, 1]),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                }
            )
        )
    )


def test_output_dict_plot_get_evaluator_lazy_false() -> None:
    assert (
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
        .get_evaluator(lazy=False)
        .equal(
            Evaluator(
                {
                    "one": {},
                    "two": {
                        "accuracy": 1.0,
                        "count": 5,
                        "count_correct": 5,
                        "count_incorrect": 0,
                        "error": 0.0,
                    },
                }
            )
        )
    )


def test_output_dict_plot_get_evaluator_empty() -> None:
    assert OutputDict({}).get_evaluator().equal(EvaluatorDict({}))


def test_output_dict_plot_get_plotter_lazy_true() -> None:
    assert (
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
        .get_plotter()
        .equal(PlotterDict({"one": Plotter(), "two": Plotter()}))
    )


def test_output_dict_plot_get_plotter_lazy_false() -> None:
    assert (
        OutputDict(
            {
                "one": Output(
                    content=ContentGenerator("meow"), evaluator=Evaluator(), plotter=Plotter()
                ),
                "two": AccuracyOutput(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            }
        )
        .get_plotter(lazy=False)
        .equal(Plotter({"one": {}, "two": {}}))
    )


def test_output_dict_plot_get_plotter_empty() -> None:
    assert OutputDict({}).get_plotter().equal(PlotterDict({}))
