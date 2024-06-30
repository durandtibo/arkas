from __future__ import annotations

from unittest.mock import patch

from arkas import is_distributed, is_main_process

#####################################
#     Tests for is_main_process     #
#####################################


def test_is_main_process_true() -> None:
    # By definition, a non-distributed process is the main process.
    assert is_main_process()


@patch("arkas.comm.get_rank", lambda: 3)
def test_is_main_process_false() -> None:
    assert not is_main_process()


####################################
#     Tests for is_distributed     #
####################################


@patch("arkas.comm.get_world_size", lambda: 3)
def test_is_distributed_true() -> None:
    assert is_distributed()


def test_is_distributed_false() -> None:
    assert not is_distributed()
