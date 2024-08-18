from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from objectory import OBJECT_TARGET

from arkas.evaluator import AccuracyEvaluator, is_evaluator_config, setup_evaluator

if TYPE_CHECKING:
    import pytest

########################################
#     Tests for is_evaluator_config     #
########################################


def test_is_evaluator_config_true() -> None:
    assert is_evaluator_config({OBJECT_TARGET: "arkas.evaluator.AccuracyEvaluator"})


def test_is_evaluator_config_false() -> None:
    assert not is_evaluator_config({OBJECT_TARGET: "collections.Counter"})


####################################
#     Tests for setup_evaluator     #
####################################


def test_setup_evaluator_object() -> None:
    evaluator = AccuracyEvaluator(y_true="target", y_pred="pred")
    assert setup_evaluator(evaluator) is evaluator


def test_setup_evaluator_dict() -> None:
    assert isinstance(
        setup_evaluator(
            {
                OBJECT_TARGET: "arkas.evaluator.AccuracyEvaluator",
                "y_true": "target",
                "y_pred": "pred",
            }
        ),
        AccuracyEvaluator,
    )


def test_setup_evaluator_incorrect_type(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(level=logging.WARNING):
        assert isinstance(setup_evaluator({OBJECT_TARGET: "collections.Counter"}), Counter)
        assert caplog.messages
