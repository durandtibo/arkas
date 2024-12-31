r"""Contain a simple evaluation runner implementation."""

from __future__ import annotations

__all__ = ["AnalysisRunner"]

import logging
from typing import TYPE_CHECKING, Any

from coola.nested import to_flat_dict
from coola.utils import str_indent, str_mapping
from coola.utils.path import sanitize_path
from grizz.ingestor import BaseIngestor, setup_ingestor
from grizz.transformer import BaseTransformer, setup_transformer
from iden.io import BaseSaver, setup_saver

from arkas.analyzer.base import BaseAnalyzer, setup_analyzer
from arkas.reporter import BaseReporter, setup_reporter
from arkas.runner.base import BaseRunner

if TYPE_CHECKING:
    from pathlib import Path

    from arkas.output import BaseOutput

logger = logging.getLogger(__name__)


class AnalysisRunner(BaseRunner):
    r"""Implement a runner to analyze data.

    Args:
        ingestor: The data ingestor or its configuration.
        transformer: The data transformer or its configuration.
        analyzer: The analyzer or its configuration.
        metric_saver: The metric saver or its configuration.
        metric_path: The path where to save the metrics.
        show_metrics: If ``True``, the metrics are shown in the
            logging output.

    Example usage:

    ```pycon

    >>> import tempfile
    >>> import polars as pl
    >>> from pathlib import Path
    >>> from iden.io import PickleSaver
    >>> from grizz.ingestor import Ingestor
    >>> from grizz.transformer import SequentialTransformer
    >>> from arkas.analyzer import AccuracyAnalyzer
    >>> from arkas.reporter import Reporter
    >>> from arkas.runner import AnalysisRunner
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     path = Path(tmpdir).joinpath("metrics.pkl")
    ...     runner = AnalysisRunner(
    ...         ingestor=Ingestor(
    ...             pl.DataFrame(
    ...                 {
    ...                     "pred": [3, 2, 0, 1, 0],
    ...                     "target": [3, 2, 0, 1, 0],
    ...                 }
    ...             )
    ...         ),
    ...         transformer=SequentialTransformer(transformers=[]),
    ...         analyzer=AccuracyAnalyzer(y_true="target", y_pred="pred"),
    ...         reporter=Reporter(),
    ...         saver=PickleSaver(),
    ...         path=path,
    ...     )
    ...     print(runner)
    ...     runner.run()
    ...
    EvaluationRunner(
      (ingestor): Ingestor(shape=(5, 2))
      (transformer): SequentialTransformer()
      (analyzer): AccuracyEvaluator(y_true='target', y_pred='pred', drop_nulls=True, nan_policy='propagate')
      (saver): PickleSaver(protocol=5)
      (path): .../metrics.pkl
      (show_metrics): True
    )

    ```
    """

    def __init__(
        self,
        ingestor: BaseIngestor | dict,
        transformer: BaseTransformer | dict,
        analyzer: BaseAnalyzer | dict,
        reporter: BaseReporter | dict,
        metric_saver: BaseSaver | dict,
        metric_path: Path | str,
        show_metrics: bool = True,
    ) -> None:
        self._ingestor = setup_ingestor(ingestor)
        self._transformer = setup_transformer(transformer)
        self._analyzer = setup_analyzer(analyzer)
        self._reporter = setup_reporter(reporter)
        self._metric_saver = setup_saver(metric_saver)
        self._metric_path = sanitize_path(metric_path)
        self._show_metrics = bool(show_metrics)

    def __repr__(self) -> str:
        args = str_indent(
            str_mapping(
                {
                    "ingestor": self._ingestor,
                    "transformer": self._transformer,
                    "analyzer": self._analyzer,
                    "reporter": self._reporter,
                    "metric_saver": self._metric_saver,
                    "metric_path": self._metric_path,
                    "show_metrics": self._show_metrics,
                }
            )
        )
        return f"{self.__class__.__qualname__}(\n  {args}\n)"

    def run(self) -> Any:
        logger.info("Ingesting data...")
        raw_data = self._ingestor.ingest()
        logger.info("Transforming data...")
        data = self._transformer.transform(raw_data)
        logger.info("Analyzing...")
        output = self._analyzer.analyze(data)
        logger.info(f"output:\n{output}")

        self._export_metrics(output)

        self._reporter.generate(output)

    def _export_metrics(self, output: BaseOutput) -> None:
        logger.info("Computing metrics...")
        metrics = output.get_evaluator().evaluate()
        logger.info(f"Saving metrics at {self._metric_path}...")
        self._metric_saver.save(metrics, path=self._metric_path, exist_ok=True)

        if self._show_metrics:
            logger.info(f"metrics:\n{str_mapping(to_flat_dict(metrics), sorted_keys=True)}")
