# arkas

<p align="center">
    <a href="https://github.com/durandtibo/arkas/actions">
        <img alt="CI" src="https://github.com/durandtibo/arkas/workflows/CI/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/arkas/actions">
        <img alt="Nightly Tests" src="https://github.com/durandtibo/arkas/workflows/Nightly%20Tests/badge.svg">
    </a>
    <a href="https://github.com/durandtibo/arkas/actions">
        <img alt="Nightly Package Tests" src="https://github.com/durandtibo/arkas/workflows/Nightly%20Package%20Tests/badge.svg">
    </a>
    <br/>
    <a href="https://durandtibo.github.io/arkas/">
        <img alt="Documentation" src="https://github.com/durandtibo/arkas/workflows/Documentation%20(stable)/badge.svg">
    </a>
    <a href="https://durandtibo.github.io/arkas/dev/">
        <img alt="Documentation" src="https://github.com/durandtibo/arkas/workflows/Documentation%20(unstable)/badge.svg">
    </a>
    <br/>
    <a href="https://codecov.io/gh/durandtibo/arkas">
        <img alt="Codecov" src="https://codecov.io/gh/durandtibo/arkas/branch/main/graph/badge.svg">
    </a>
    <a href="https://codeclimate.com/github/durandtibo/arkas/maintainability">
        <img src="https://api.codeclimate.com/v1/badges/bc15147bbdd318137184/maintainability" />
    </a>
    <a href="https://codeclimate.com/github/durandtibo/arkas/test_coverage">
        <img src="https://api.codeclimate.com/v1/badges/bc15147bbdd318137184/test_coverage" />
    </a>
    <br/>
    <a href="https://github.com/psf/black">
        <img  alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
    </a>
    <a href="https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/%20style-google-3666d6.svg">
    </a>
    <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff" style="max-width:100%;">
    </a>
    <a href="https://github.com/guilatrova/tryceratops">
        <img  alt="Doc style: google" src="https://img.shields.io/badge/try%2Fexcept%20style-tryceratops%20%F0%9F%A6%96%E2%9C%A8-black">
    </a>
    <br/>
    <a href="https://pypi.org/project/arkas/">
        <img alt="PYPI version" src="https://img.shields.io/pypi/v/arkas">
    </a>
    <a href="https://pypi.org/project/arkas/">
        <img alt="Python" src="https://img.shields.io/pypi/pyversions/arkas.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-3-Clause">
        <img alt="BSD-3-Clause" src="https://img.shields.io/pypi/l/arkas">
    </a>
    <br/>
    <a href="https://pepy.tech/project/arkas">
        <img  alt="Downloads" src="https://static.pepy.tech/badge/arkas">
    </a>
    <a href="https://pepy.tech/project/arkas">
        <img  alt="Monthly downloads" src="https://static.pepy.tech/badge/arkas/month">
    </a>
    <br/>
</p>

## Overview

`arkas` is a simple Python library to evaluate ML model performances.

## Installation

We highly recommend installing
a [virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).
`arkas` can be installed from pip using the following command:

```shell
pip install arkas
```

To make the package as slim as possible, only the minimal packages required to use `arkas` are
installed.
To include all the dependencies, you can use the following command:

```shell
pip install arkas[all]
```

Please check the [get started page](https://durandtibo.github.io/arkas/get_started) to see how to
install only some specific dependencies or other alternatives to install the library.
The following is the corresponding `arkas` versions and tested dependencies.

| `arkas` | `coola`        | `grizz`      | `iden`       | `matplotlib` | `numpy`       | `objectory`  | `polars`     | `scikit-learn` | `python`      |
|---------|----------------|--------------|--------------|--------------|---------------|--------------|--------------|----------------|---------------|
| `main`  | `>=0.8.2,<1.0` | `>=0.1,<1.0` | `>=0.1,<1.0` | `>=3.6,<4.0` | `>=1.22,<2.0` | `>=0.2,<1.0` | `>=1.0,<2.0` | `>=1.3,<2.0`   | `>=3.9,<3.13` |

| `arkas` | `colorlog`<sup>*</sup> | `hya`<sup>*</sup> | `markdown`<sup>*</sup> | `plotly`<sup>*</sup> | `scipy`<sup>*</sup> | `tqdm`<sup>*</sup> |
|---------|------------------------|-------------------|------------------------|----------------------|---------------------|--------------------|
| `main`  | `>=6.7,<7.0`           | `>=0.2,<1.0`      | `>=3.4,<4.0`           | `>=5.24.0,<6.0`      | `>=1.10,<2.0`       | `>=4.65,<5.0`      |

<sup>*</sup> indicates an optional dependency

## Contributing

Please check the instructions in [CONTRIBUTING.md](.github/CONTRIBUTING.md).

## Suggestions and Communication

Everyone is welcome to contribute to the community.
If you have any questions or suggestions, you can
submit [Github Issues](https://github.com/durandtibo/arkas/issues).
We will reply to you as soon as possible. Thank you very much.

## API stability

:warning: While `arkas` is in development stage, no API is guaranteed to be stable from one
release to the next.
In fact, it is very likely that the API will change multiple times before a stable 1.0.0 release.
In practice, this means that upgrading `arkas` to a new version will possibly break any code that
was using the old version of `arkas`.

## License

`arkas` is licensed under BSD 3-Clause "New" or "Revised" license available in [LICENSE](LICENSE)
file.
