from __future__ import annotations

from typing import Any

from coola import objects_are_equal

from arkas.evaluator2 import BaseCacheEvaluator, BaseEvaluator


class MyCacheEvaluator(BaseCacheEvaluator):

    def compute(self) -> BaseEvaluator:
        return self

    def equal(self, other: Any, equal_nan: bool = False) -> bool:  # noqa: ARG002
        return isinstance(other, self.__class__)

    def _evaluate(self) -> dict:
        return {"metric1": 0.42, "metric2": 1.2}


########################################
#     Tests for BaseCacheEvaluator     #
########################################


def test_base_cache_evaluator_evaluate() -> None:
    evaluator = MyCacheEvaluator()
    out = evaluator.evaluate()
    assert objects_are_equal(out, {"metric1": 0.42, "metric2": 1.2})
    assert objects_are_equal(evaluator._cached_metrics, {"metric1": 0.42, "metric2": 1.2})
    assert evaluator._cached_metrics is not out


def test_base_cache_evaluator_evaluate_multi() -> None:
    evaluator = MyCacheEvaluator()
    out1 = evaluator.evaluate()
    out2 = evaluator.evaluate()
    assert objects_are_equal(out1, out2)
    assert out1 is not out2


def test_base_cache_evaluator_evaluate_prefix_suffix() -> None:
    evaluator = MyCacheEvaluator()
    out = evaluator.evaluate(prefix="prefix_", suffix="_suffix")
    assert objects_are_equal(out, {"prefix_metric1_suffix": 0.42, "prefix_metric2_suffix": 1.2})
    assert objects_are_equal(evaluator._cached_metrics, {"metric1": 0.42, "metric2": 1.2})
