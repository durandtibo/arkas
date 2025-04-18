from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from arkas.content import (
    AccuracyContentGenerator,
    BaseContentGenerator,
    ContentGenerator,
    ContentGeneratorDict,
)
from arkas.state import AccuracyState


@pytest.fixture
def generators() -> dict[str, BaseContentGenerator]:
    return {
        "one": ContentGenerator("meow"),
        "two": ContentGenerator(),
        "three": ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
                "two": ContentGenerator(),
                "three": ContentGeneratorDict(
                    {
                        "one": ContentGenerator("meow"),
                        "two": ContentGenerator(),
                    }
                ),
            }
        ),
    }


##########################################
#     Tests for ContentGeneratorDict     #
##########################################


def test_content_generator_dict_repr() -> None:
    assert repr(ContentGeneratorDict({})).startswith("ContentGeneratorDict(")


def test_content_generator_dict_str() -> None:
    assert str(ContentGeneratorDict({})).startswith("ContentGeneratorDict(")


def test_content_generator_dict_compute() -> None:
    assert (
        ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
                "two": AccuracyContentGenerator.from_state(
                    AccuracyState(
                        y_true=np.array([1, 0, 0, 1, 1]),
                        y_pred=np.array([1, 0, 0, 1, 1]),
                        y_true_name="target",
                        y_pred_name="pred",
                    )
                ),
            },
            max_toc_depth=4,
        )
        .compute()
        .equal(
            ContentGeneratorDict(
                {
                    "one": ContentGenerator("meow"),
                    "two": ContentGenerator(
                        "<ul>\n"
                        "  <li><b>accuracy</b>: 1.0000 (5/5)</li>\n"
                        "  <li><b>error</b>: 0.0000 (0/5)</li>\n"
                        "  <li><b>number of samples</b>: 5</li>\n"
                        "  <li><b>target label column</b>: target</li>\n"
                        "  <li><b>predicted label column</b>: pred</li>\n"
                        "</ul>"
                    ),
                },
                max_toc_depth=4,
            )
        )
    )


def test_content_generator_dict_equal_true() -> None:
    assert ContentGeneratorDict(
        {
            "one": ContentGenerator("meow"),
            "two": ContentGenerator(),
        }
    ).equal(
        ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
                "two": ContentGenerator(),
            }
        )
    )


def test_content_generator_dict_equal_false_different_content_generators() -> None:
    assert not ContentGeneratorDict(
        {
            "one": ContentGenerator("meow"),
            "two": ContentGenerator(),
        }
    ).equal(
        ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
            }
        )
    )


def test_content_generator_dict_equal_false_different_types() -> None:
    assert not ContentGeneratorDict(
        {
            "one": ContentGenerator("meow"),
            "two": ContentGenerator(),
        }
    ).equal(ContentGenerator("meow"))


def test_content_generator_dict_equal_nan_true() -> None:
    assert ContentGeneratorDict(
        {
            "one": ContentGenerator("meow"),
            "two": ContentGenerator(),
        }
    ).equal(
        ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
                "two": ContentGenerator(),
            }
        ),
        equal_nan=True,
    )


def test_content_generator_dict_equal_nan_false() -> None:
    assert ContentGeneratorDict(
        {
            "one": ContentGenerator("meow"),
            "two": ContentGenerator(),
        }
    ).equal(
        ContentGeneratorDict(
            {
                "one": ContentGenerator("meow"),
                "two": ContentGenerator(),
            }
        ),
    )


def test_content_generator_dict_generate_body(generators: dict[str, BaseContentGenerator]) -> None:
    assert isinstance(
        ContentGeneratorDict(generators).generate_body(),
        str,
    )


@pytest.mark.parametrize("max_toc_depth", [0, 1, 2])
def test_content_generator_dict_generate_body_args(
    generators: dict[str, BaseContentGenerator], max_toc_depth: int
) -> None:
    assert isinstance(
        ContentGeneratorDict(generators=generators, max_toc_depth=max_toc_depth).generate_body(
            number="1.", tags=["meow"]
        ),
        str,
    )


def test_content_generator_dict_generate_body_empty() -> None:
    assert isinstance(ContentGeneratorDict({}).generate_body(), str)


def test_content_generator_dict_generate_toc(generators: dict[str, BaseContentGenerator]) -> None:
    assert isinstance(
        ContentGeneratorDict(generators).generate_toc(),
        str,
    )


def test_content_generator_dict_generate_toc_args(
    generators: dict[str, BaseContentGenerator],
) -> None:
    assert isinstance(
        ContentGeneratorDict(generators).generate_toc(number="1.", tags=["meow"]),
        str,
    )


def test_content_generator_dict_generate_toc_too_deep(
    generators: dict[str, BaseContentGenerator],
) -> None:
    assert isinstance(
        ContentGeneratorDict(generators).generate_toc(max_depth=2, depth=2),
        str,
    )


def test_content_generator_dict_generate_toc_empty() -> None:
    assert isinstance(ContentGeneratorDict({}).generate_toc(), str)


def test_content_generator_dict_generate_toc_empty_toc() -> None:
    assert isinstance(
        ContentGeneratorDict(
            {
                "one": Mock(spec=BaseContentGenerator, generate_toc=Mock(return_value="")),
                "two": Mock(spec=BaseContentGenerator, generate_toc=Mock(return_value="")),
            }
        ).generate_toc(),
        str,
    )
