from __future__ import annotations

import numpy as np
import polars as pl

from arkas.analyzer import AccuracyAnalyzer, BalancedAccuracyAnalyzer, MappingAnalyzer
from arkas.output import AccuracyOutput, BalancedAccuracyOutput, OutputDict
from arkas.state import AccuracyState

#####################################
#     Tests for MappingAnalyzer     #
#####################################


def test_mapping_analyzer_repr() -> None:
    assert repr(
        MappingAnalyzer(
            {
                "one": AccuracyAnalyzer(y_true="target", y_pred="pred"),
                "two": BalancedAccuracyAnalyzer(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("MappingAnalyzer(")


def test_mapping_analyzer_str() -> None:
    assert str(
        MappingAnalyzer(
            {
                "one": AccuracyAnalyzer(y_true="target", y_pred="pred"),
                "two": BalancedAccuracyAnalyzer(y_true="target", y_pred="pred"),
            }
        )
    ).startswith("MappingAnalyzer(")


def test_mapping_analyzer_analyze() -> None:
    assert (
        MappingAnalyzer(
            {
                "one": AccuracyAnalyzer(y_true="target", y_pred="pred"),
                "two": BalancedAccuracyAnalyzer(y_true="target", y_pred="pred"),
            }
        )
        .analyze(pl.DataFrame({"pred": [1, 0, 0, 1, 1], "target": [1, 0, 1, 0, 1]}))
        .equal(
            OutputDict(
                {
                    "one": AccuracyOutput(
                        state=AccuracyState(
                            y_true=np.array([1, 0, 1, 0, 1]),
                            y_pred=np.array([1, 0, 0, 1, 1]),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                    "two": BalancedAccuracyOutput(
                        state=AccuracyState(
                            y_true=np.array([1, 0, 1, 0, 1]),
                            y_pred=np.array([1, 0, 0, 1, 1]),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                }
            )
        )
    )


def test_mapping_analyzer_analyze_empty() -> None:
    assert (
        MappingAnalyzer(
            {
                "one": AccuracyAnalyzer(y_true="target", y_pred="pred"),
                "two": BalancedAccuracyAnalyzer(y_true="target", y_pred="pred"),
            }
        )
        .analyze(
            pl.DataFrame({"pred": [], "target": []}, schema={"pred": pl.Int64, "target": pl.Int64})
        )
        .equal(
            OutputDict(
                {
                    "one": AccuracyOutput(
                        state=AccuracyState(
                            y_true=np.array([], dtype=int),
                            y_pred=np.array([], dtype=int),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                    "two": BalancedAccuracyOutput(
                        state=AccuracyState(
                            y_true=np.array([], dtype=int),
                            y_pred=np.array([], dtype=int),
                            y_true_name="target",
                            y_pred_name="pred",
                        )
                    ),
                }
            )
        )
    )
