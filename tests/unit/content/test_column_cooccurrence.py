from __future__ import annotations

import numpy as np

from arkas.content import ColumnCooccurrenceContentGenerator, ContentGenerator
from arkas.content.column_cooccurrence import (
    create_table,
    create_table_row,
    create_table_section,
    create_template,
)
from arkas.state import ColumnCooccurrenceState

########################################################
#     Tests for ColumnCooccurrenceContentGenerator     #
########################################################


def test_column_cooccurrence_content_generator_repr() -> None:
    assert repr(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("ColumnCooccurrenceContentGenerator(")


def test_column_cooccurrence_content_generator_str() -> None:
    assert str(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    ).startswith("ColumnCooccurrenceContentGenerator(")


def test_column_cooccurrence_content_generator_compute() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).compute(),
        ContentGenerator,
    )


def test_column_cooccurrence_content_generator_equal_true() -> None:
    assert ColumnCooccurrenceContentGenerator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_column_cooccurrence_content_generator_equal_false_different_state() -> None:
    assert not ColumnCooccurrenceContentGenerator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.ones((3, 3)),
                columns=["col1", "col2", "col3"],
            )
        )
    )


def test_column_cooccurrence_content_generator_equal_false_different_type() -> None:
    assert not ColumnCooccurrenceContentGenerator(
        ColumnCooccurrenceState(
            matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
            columns=["col1", "col2", "col3"],
        )
    ).equal(42)


def test_column_cooccurrence_content_generator_generate_content() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).generate_content(),
        str,
    )


def test_column_cooccurrence_content_generator_generate_content_empty() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(matrix=np.zeros((0, 0)), columns=[])
        ).generate_content(),
        str,
    )


def test_column_cooccurrence_content_generator_generate_body() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).generate_body(),
        str,
    )


def test_column_cooccurrence_content_generator_generate_body_args() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).generate_body(number="1.", tags=["meow"], depth=1),
        str,
    )


def test_column_cooccurrence_content_generator_generate_toc() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).generate_toc(),
        str,
    )


def test_column_cooccurrence_content_generator_generate_toc_args() -> None:
    assert isinstance(
        ColumnCooccurrenceContentGenerator(
            ColumnCooccurrenceState(
                matrix=np.array([[3, 2, 1], [2, 3, 1], [1, 1, 3]], dtype=int),
                columns=["col1", "col2", "col3"],
            )
        ).generate_toc(number="1.", tags=["meow"], depth=1),
        str,
    )


#####################################
#     Tests for create_template     #
#####################################


def test_create_template() -> None:
    assert isinstance(create_template(), str)


##########################################
#     Tests for create_table_section     #
##########################################


def test_create_table_section() -> None:
    assert isinstance(
        create_table_section(
            matrix=np.array([[5, 7, 1], [0, 6, 3], [8, 2, 4]]), columns=["col1", "col2", "col3"]
        ),
        str,
    )


def test_create_table_section_empty() -> None:
    assert isinstance(create_table_section(matrix=np.zeros((0, 0)), columns=[]), str)


##################################
#     Tests for create_table     #
##################################


def test_create_table() -> None:
    assert create_table(
        np.array([[80, 4, 6], [4, 2, 5], [6, 5, 3]]), columns=["col1", "col2", "col3"]
    ) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        '        <tr><th>1</th><td>col1</td><td>col1</td><td style="text-align: right; font-variant-numeric: tabular-nums;">80</td><td style="text-align: right; font-variant-numeric: tabular-nums;">80.0000 %</td></tr>\n'
        '        <tr><th>2</th><td>col1</td><td>col3</td><td style="text-align: right; font-variant-numeric: tabular-nums;">6</td><td style="text-align: right; font-variant-numeric: tabular-nums;">6.0000 %</td></tr>\n'
        '        <tr><th>3</th><td>col2</td><td>col3</td><td style="text-align: right; font-variant-numeric: tabular-nums;">5</td><td style="text-align: right; font-variant-numeric: tabular-nums;">5.0000 %</td></tr>\n'
        '        <tr><th>4</th><td>col1</td><td>col2</td><td style="text-align: right; font-variant-numeric: tabular-nums;">4</td><td style="text-align: right; font-variant-numeric: tabular-nums;">4.0000 %</td></tr>\n'
        '        <tr><th>5</th><td>col3</td><td>col3</td><td style="text-align: right; font-variant-numeric: tabular-nums;">3</td><td style="text-align: right; font-variant-numeric: tabular-nums;">3.0000 %</td></tr>\n'
        '        <tr><th>6</th><td>col2</td><td>col2</td><td style="text-align: right; font-variant-numeric: tabular-nums;">2</td><td style="text-align: right; font-variant-numeric: tabular-nums;">2.0000 %</td></tr>\n'
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


def test_create_table_top_2() -> None:
    assert create_table(
        np.array([[80, 4, 6], [4, 2, 5], [6, 5, 3]]), columns=["col1", "col2", "col3"], top=2
    ) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        '        <tr><th>1</th><td>col1</td><td>col1</td><td style="text-align: right; font-variant-numeric: tabular-nums;">80</td><td style="text-align: right; font-variant-numeric: tabular-nums;">80.0000 %</td></tr>\n'
        '        <tr><th>2</th><td>col1</td><td>col3</td><td style="text-align: right; font-variant-numeric: tabular-nums;">6</td><td style="text-align: right; font-variant-numeric: tabular-nums;">6.0000 %</td></tr>\n'
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


def test_create_table_empty() -> None:
    assert create_table(matrix=np.zeros((0, 0)), columns=[]) == (
        '<table class="table table-hover table-responsive w-auto" >\n'
        '    <thead class="thead table-group-divider">\n'
        "        <tr><th>rank</th><th>column 1</th><th>column 2</th><th>count</th><th>percentage</th></tr>\n"
        "    </thead>\n"
        '    <tbody class="tbody table-group-divider">\n'
        "        \n"
        '        <tr class="table-group-divider"></tr>\n'
        "    </tbody>\n"
        "</table>"
    )


######################################
#     Tests for create_table_row     #
######################################


def test_create_table_row() -> None:
    assert create_table_row(rank=2, col1="cat", col2="meow", count=42, total=100) == (
        '<tr><th>2</th><td>cat</td><td>meow</td><td style="text-align: right; '
        'font-variant-numeric: tabular-nums;">42</td>'
        '<td style="text-align: right; font-variant-numeric: tabular-nums;">42.0000 %</td></tr>'
    )


def test_create_table_row_empty() -> None:
    assert create_table_row(rank=2, col1="cat", col2="meow", count=0, total=0) == (
        '<tr><th>2</th><td>cat</td><td>meow</td><td style="text-align: right; '
        'font-variant-numeric: tabular-nums;">0</td>'
        '<td style="text-align: right; font-variant-numeric: tabular-nums;">nan %</td></tr>'
    )
