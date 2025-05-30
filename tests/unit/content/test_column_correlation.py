from __future__ import annotations

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.content import ColumnCorrelationContentGenerator, ContentGenerator
from arkas.content.column_correlation import (
    create_table,
    create_table_row,
    create_template,
    sort_metrics,
)
from arkas.evaluator2 import ColumnCorrelationEvaluator
from arkas.state import TargetDataFrameState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "col2": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "col3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        },
        schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
    )


#######################################################
#     Tests for ColumnCorrelationContentGenerator     #
#######################################################


def test_column_correlation_content_generator_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        )
    ).startswith("ColumnCorrelationContentGenerator(")


def test_column_correlation_content_generator_str(dataframe: pl.DataFrame) -> None:
    assert str(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        )
    ).startswith("ColumnCorrelationContentGenerator(")


def test_column_correlation_content_generator_compute(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).compute(),
        ContentGenerator,
    )


def test_column_correlation_content_generator_equal_true(dataframe: pl.DataFrame) -> None:
    assert ColumnCorrelationContentGenerator(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    ).equal(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        )
    )


def test_column_correlation_content_generator_equal_false_different_evaluator(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCorrelationContentGenerator(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    ).equal(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col1"))
        )
    )


def test_column_correlation_content_generator_equal_false_different_type(
    dataframe: pl.DataFrame,
) -> None:
    assert not ColumnCorrelationContentGenerator(
        ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
    ).equal(42)


def test_column_correlation_content_generator_generate_content(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).generate_content(),
        str,
    )


def test_column_correlation_content_generator_generate_content_empty_rows() -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(
                TargetDataFrameState(
                    pl.DataFrame(
                        {"col1": [], "col2": [], "col3": []},
                        schema={"col1": pl.Float64, "col2": pl.Float64, "col3": pl.Float64},
                    ),
                    target_column="col3",
                )
            )
        ).generate_content(),
        str,
    )


def test_column_correlation_content_generator_generate_body(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).generate_body(),
        str,
    )


def test_column_correlation_content_generator_generate_body_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_column_correlation_content_generator_generate_toc(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).generate_toc(),
        str,
    )


def test_column_correlation_content_generator_generate_toc_args(dataframe: pl.DataFrame) -> None:
    assert isinstance(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_column_correlation_content_generator_from_state(dataframe: pl.DataFrame) -> None:
    assert ColumnCorrelationContentGenerator.from_state(
        TargetDataFrameState(dataframe, target_column="col3")
    ).equal(
        ColumnCorrelationContentGenerator(
            ColumnCorrelationEvaluator(TargetDataFrameState(dataframe, target_column="col3"))
        )
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert isinstance(
        create_table(
            metrics={
                "col1": {
                    "count": 7,
                    "pearson_coeff": 1.0,
                    "pearson_pvalue": 0.0,
                    "spearman_coeff": 1.0,
                    "spearman_pvalue": 0.0,
                },
                "col2": {
                    "count": 7,
                    "pearson_coeff": -1.0,
                    "pearson_pvalue": 0.0,
                    "spearman_coeff": -1.0,
                    "spearman_pvalue": 0.0,
                },
            },
        ),
        str,
    )


def test_create_table_empty() -> None:
    assert isinstance(create_table(metrics={}), str)


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert isinstance(
        create_table_row(
            column="col1",
            metrics={
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
        ),
        str,
    )


def test_create_table_row_empty() -> None:
    assert isinstance(create_table_row(column="col1", metrics={}), str)


##################################
#     Tests for sort_metrics     #
##################################


def test_sort_metrics() -> None:
    metrics = sort_metrics(
        {
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
        }
    )
    assert list(metrics.keys()) == ["col3", "col1", "col2"]
    assert objects_are_equal(
        metrics,
        {
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
        },
    )


def test_sort_metrics_key() -> None:
    metrics = sort_metrics(
        {
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
        },
        key="pearson_coeff",
    )
    assert list(metrics.keys()) == ["col2", "col3", "col1"]
    assert objects_are_equal(
        metrics,
        {
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
        },
    )


def test_sort_metrics_with_nan() -> None:
    metrics = sort_metrics(
        {
            "col0": {
                "count": 0,
                "pearson_coeff": float("nan"),
                "pearson_pvalue": float("nan"),
                "spearman_coeff": float("nan"),
                "spearman_pvalue": float("nan"),
            },
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
        }
    )
    assert list(metrics.keys()) == ["col3", "col1", "col2", "col0"]
    assert objects_are_equal(
        metrics,
        {
            "col3": {
                "count": 7,
                "pearson_coeff": 0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 1.0,
                "spearman_pvalue": 0.0,
            },
            "col1": {
                "count": 7,
                "pearson_coeff": -0.5,
                "pearson_pvalue": 0.0,
                "spearman_coeff": 0.5,
                "spearman_pvalue": 0.0,
            },
            "col2": {
                "count": 7,
                "pearson_coeff": 1.0,
                "pearson_pvalue": 0.0,
                "spearman_coeff": -1.0,
                "spearman_pvalue": 0.0,
            },
            "col0": {
                "count": 0,
                "pearson_coeff": float("nan"),
                "pearson_pvalue": float("nan"),
                "spearman_coeff": float("nan"),
                "spearman_pvalue": float("nan"),
            },
        },
        equal_nan=True,
    )
