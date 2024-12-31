r"""Contain the base class to implement an output."""

from __future__ import annotations

__all__ = ["BaseOutput"]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from arkas.evaluator2.base import BaseEvaluator
    from arkas.hcg.base import BaseContentGenerator
    from arkas.plotter.base import BasePlotter


class BaseOutput(ABC):
    r"""Define the base class to implement an output.

    Example usage:

    ```pycon

    >>> import numpy as np
    >>> from arkas.output import AccuracyOutput
    >>> from arkas.state import AccuracyState
    >>> output = AccuracyOutput(
    ...     AccuracyState(
    ...         y_true=np.array([1, 0, 0, 1, 1]),
    ...         y_pred=np.array([1, 0, 0, 1, 1]),
    ...         y_true_name="target",
    ...         y_pred_name="pred",
    ...     )
    ... )
    >>> output
    AccuracyOutput(
      (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
      (nan_policy): propagate
    )

    ```
    """

    @abstractmethod
    def equal(self, other: Any, equal_nan: bool = False) -> bool:
        r"""Indicate if two outputs are equal or not.

        Args:
            other: The other output to compare.
            equal_nan: Whether to compare NaN's as equal. If ``True``,
                NaN's in both objects will be considered equal.

        Returns:
            ``True`` if the two outputs are equal, otherwise ``False``.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.output import AccuracyOutput
        >>> from arkas.state import AccuracyState
        >>> output1 = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output2 = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output3 = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 0, 0]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output1.equal(output2)
        True
        >>> output1.equal(output3)
        False

        ```
        """

    @abstractmethod
    def get_content_generator(self) -> BaseContentGenerator:
        r"""Get the HTML content generator associated to the output.

        Returns:
            The HTML content generator.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.output import AccuracyOutput
        >>> from arkas.state import AccuracyState
        >>> output = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output.get_content_generator()
        AccuracyContentGenerator(
          (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
          (nan_policy): propagate
        )

        ```
        """

    @abstractmethod
    def get_evaluator(self, lazy: bool = True) -> BaseEvaluator:
        r"""Get the evaluator associated to the output.

        Args:
            lazy: If ``True``, it forces the computation of the
                metrics, otherwise it returns an evaluator object
                that contains the logic to evaluate the metrics.

        Returns:
            The evaluator.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.output import AccuracyOutput
        >>> from arkas.state import AccuracyState
        >>> output = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output.get_evaluator()
        AccuracyEvaluator(
          (state): AccuracyState(y_true=(5,), y_pred=(5,), y_true_name='target', y_pred_name='pred')
          (nan_policy): propagate
        )

        ```
        """

    @abstractmethod
    def get_plotter(self, lazy: bool = True) -> BasePlotter:
        r"""Get the plotter associated to the output.

        Args:
            lazy: If ``True``, it forces the computation of the
                figures, otherwise it returns a plotter object
                that contains the logic to generate the figures.

        Returns:
            The plotter.

        Example usage:

        ```pycon

        >>> import numpy as np
        >>> from arkas.output import AccuracyOutput
        >>> from arkas.state import AccuracyState
        >>> output = AccuracyOutput(
        ...     AccuracyState(
        ...         y_true=np.array([1, 0, 0, 1, 1]),
        ...         y_pred=np.array([1, 0, 0, 1, 1]),
        ...         y_true_name="target",
        ...         y_pred_name="pred",
        ...     )
        ... )
        >>> output.get_plotter()
        Plotter(count=0)

        ```
        """
