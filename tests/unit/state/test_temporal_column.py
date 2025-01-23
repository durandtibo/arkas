from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest
from coola import objects_are_equal

from arkas.figure import MatplotlibFigureConfig
from arkas.state import TemporalColumnState


@pytest.fixture
def dataframe() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "col1": [0, 1, 1, 0, 0, 1, 0],
            "col2": [0, 1, 0, 1, 0, 1, 0],
            "col3": [0, 0, 0, 0, 1, 1, 1],
            "datetime": [
                datetime(year=2020, month=1, day=1, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=2, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=3, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=4, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=5, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=6, tzinfo=timezone.utc),
                datetime(year=2020, month=1, day=7, tzinfo=timezone.utc),
            ],
        },
        schema={
            "col1": pl.Int64,
            "col2": pl.Int64,
            "col3": pl.Int64,
            "datetime": pl.Datetime(time_unit="us", time_zone="UTC"),
        },
    )


#########################################
#     Tests for TemporalColumnState     #
#########################################


def test_temporal_column_state_init_target_column_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'col' is not in the DataFrame:"):
        TemporalColumnState(dataframe, target_column="col", temporal_column="datetime")


def test_temporal_column_state_init_temporal_column_incorrect(dataframe: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="The column 'time' is not in the DataFrame:"):
        TemporalColumnState(dataframe, target_column="col2", temporal_column="time")


def test_temporal_column_state_dataframe(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").dataframe,
        dataframe,
    )


def test_temporal_column_state_target_column(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime"
        ).target_column
        == "col2"
    )


def test_temporal_column_state_temporal_column(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime"
        ).temporal_column
        == "datetime"
    )


def test_temporal_column_state_period(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", period="1mo"
        ).period
        == "1mo"
    )


def test_temporal_column_state_period_default(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").period
        is None
    )


def test_temporal_column_state_nan_policy(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", nan_policy="raise"
        ).nan_policy
        == "raise"
    )


def test_temporal_column_state_nan_policy_default(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").nan_policy
        == "propagate"
    )


def test_temporal_column_state_figure_config(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            figure_config=MatplotlibFigureConfig(dpi=300),
        ).figure_config,
        MatplotlibFigureConfig(dpi=300),
    )


def test_temporal_column_state_figure_config_default(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime"
        ).figure_config,
        MatplotlibFigureConfig(),
    )


def test_temporal_column_state_repr(dataframe: pl.DataFrame) -> None:
    assert repr(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).startswith("TemporalColumnState(")


def test_temporal_column_state_str(dataframe: pl.DataFrame) -> None:
    assert str(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    ).startswith("TemporalColumnState(")


def test_temporal_column_state_clone(dataframe: pl.DataFrame) -> None:
    state = TemporalColumnState(
        dataframe,
        target_column="col2",
        temporal_column="datetime",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone()
    assert state is not cloned_state
    assert state.equal(cloned_state)


def test_temporal_column_state_clone_deep(dataframe: pl.DataFrame) -> None:
    state = TemporalColumnState(
        dataframe,
        target_column="col2",
        temporal_column="datetime",
        period="1mo",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone()

    assert state.equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            period="1mo",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert cloned_state.equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            period="1mo",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert state.dataframe is not cloned_state.dataframe


def test_temporal_column_state_clone_shallow(dataframe: pl.DataFrame) -> None:
    state = TemporalColumnState(
        dataframe,
        target_column="col2",
        temporal_column="datetime",
        period="1mo",
        nan_policy="raise",
        figure_config=MatplotlibFigureConfig(xscale="linear"),
        column="col",
    )
    cloned_state = state.clone(deep=False)

    assert state.equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            period="1mo",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert cloned_state.equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            period="1mo",
            nan_policy="raise",
            figure_config=MatplotlibFigureConfig(xscale="linear"),
            column="col",
        )
    )
    assert state.dataframe is cloned_state.dataframe


def test_temporal_column_state_equal_true(dataframe: pl.DataFrame) -> None:
    assert TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").equal(
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime")
    )


def test_temporal_column_state_equal_false_different_dataframe(dataframe: pl.DataFrame) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalColumnState(
            pl.DataFrame({"col2": [], "datetime": []}),
            target_column="col2",
            temporal_column="datetime",
        )
    )


def test_temporal_column_state_equal_false_different_target_column(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(TemporalColumnState(dataframe, target_column="col1", temporal_column="datetime"))


def test_temporal_column_state_equal_false_different_temporal_column(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(TemporalColumnState(dataframe, target_column="col2", temporal_column="col1"))


def test_temporal_column_state_equal_false_different_period(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime", period="2mo"
    ).equal(
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", period="1mo"
        )
    )


def test_temporal_column_state_equal_false_different_nan_policy(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", nan_policy="raise"
        )
    )


def test_temporal_column_state_equal_false_different_figure_config(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalColumnState(
            dataframe,
            target_column="col2",
            temporal_column="datetime",
            figure_config=MatplotlibFigureConfig(dpi=300),
        )
    )


def test_temporal_column_state_equal_false_different_kwargs(
    dataframe: pl.DataFrame,
) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", column="col"
        )
    )


def test_temporal_column_state_equal_false_different_type(dataframe: pl.DataFrame) -> None:
    assert not TemporalColumnState(
        dataframe, target_column="col2", temporal_column="datetime"
    ).equal(42)


def test_temporal_column_state_get_arg(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", column="col"
        ).get_arg("column")
        == "col"
    )


def test_temporal_column_state_get_arg_missing(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").get_arg(
            "x"
        )
        is None
    )


def test_temporal_column_state_get_arg_missing_default(dataframe: pl.DataFrame) -> None:
    assert (
        TemporalColumnState(dataframe, target_column="col2", temporal_column="datetime").get_arg(
            "x", 42
        )
        == 42
    )


def test_temporal_column_state_get_args(dataframe: pl.DataFrame) -> None:
    assert objects_are_equal(
        TemporalColumnState(
            dataframe, target_column="col2", temporal_column="datetime", period="1mo", column="col"
        ).get_args(),
        {
            "dataframe": dataframe,
            "target_column": "col2",
            "temporal_column": "datetime",
            "period": "1mo",
            "nan_policy": "propagate",
            "figure_config": MatplotlibFigureConfig(),
            "column": "col",
        },
    )
