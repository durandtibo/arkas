r"""Contain utility functions for scatter plot."""

from __future__ import annotations

__all__ = ["find_marker_size_from_size"]


def find_marker_size_from_size(
    n: int,
    min_size: tuple[float, int] = (10.0, 100_000),
    max_size: tuple[float, int] = (32.0, 1_000),
) -> float:
    r"""Find a good marker size from the number of data points.

    Args:
        n: The number of data points.
        min_size: A tuple that indicates the minimal marker size and
            its associated number of data points. This marker size
            (first item in the tuple) is  used if the number of data
            points is greater or equal than the second item in the
            tuple.
        max_size: A tuple that indicates the maximal marker size and
            its associated number of data points. This marker size
            (first item in the tuple) is  used if the number of data
            points is lower or equal than the second item in the
            tuple.

    Returns:
        The marker size.

    Example usage:

    ```pycon

    >>> from arkas.plot.utils.scatter import find_marker_size_from_size
    >>> find_marker_size_from_size(n=1_000)
    32.0

    ```
    """
    a = (min_size[0] - max_size[0]) / (min_size[1] - max_size[1])
    b = min_size[0] - a * min_size[1]
    return max(min_size[0], min(max_size[0], a * n + b))
