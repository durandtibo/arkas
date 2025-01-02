from arkas.figure import MatplotlibFigureConfig

############################################
#     Tests for MatplotlibFigureConfig     #
############################################


def test_matplotlib_figure_config_backend() -> None:
    assert MatplotlibFigureConfig.backend() == "matplotlib"


def test_matplotlib_figure_config_repr() -> None:
    assert repr(MatplotlibFigureConfig()) == "MatplotlibFigureConfig()"


def test_matplotlib_figure_config_str() -> None:
    assert str(MatplotlibFigureConfig()) == "MatplotlibFigureConfig()"


def test_matplotlib_figure_config_equal_true() -> None:
    assert MatplotlibFigureConfig().equal(MatplotlibFigureConfig())


def test_matplotlib_figure_config_equal_false_different_kwargs() -> None:
    assert not MatplotlibFigureConfig().equal(MatplotlibFigureConfig(dpi=300))


def test_matplotlib_figure_config_equal_false_different_type() -> None:
    assert not MatplotlibFigureConfig().equal(42)


def test_matplotlib_figure_config_get_args() -> None:
    assert MatplotlibFigureConfig(dpi=300).get_args() == {"dpi": 300}
