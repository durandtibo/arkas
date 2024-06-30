r"""Contain some primitives for distributed communication."""

from __future__ import annotations

__all__ = [
    "Backend",
    "UnknownBackendError",
    "all_gather",
    "all_reduce",
    "available_backends",
    "backend",
    "barrier",
    "broadcast",
    "device",
    "finalize",
    "get_local_rank",
    "get_nnodes",
    "get_node_rank",
    "get_nproc_per_node",
    "get_rank",
    "get_world_size",
    "hostname",
    "initialize",
    "is_distributed",
    "is_main_process",
    "model_name",
    "set_local_rank",
    "show_config",
]

import logging

from ignite.distributed import utils

logger = logging.getLogger(__name__)


class Backend:
    r"""Define the name of the distributed backends currently
    supported."""

    NCCL = "nccl"
    GLOO = "gloo"


# Do not use ignite directly because it will give more freedom if we want to change one day.
# Only this file should call directly the PyTorch Ignite functions.
all_gather = utils.all_gather
all_reduce = utils.all_reduce
available_backends = utils.available_backends
backend = utils.backend
barrier = utils.barrier
broadcast = utils.broadcast
device = utils.device
finalize = utils.finalize
get_local_rank = utils.get_local_rank
get_nnodes = utils.get_nnodes
get_node_rank = utils.get_node_rank
get_nproc_per_node = utils.get_nproc_per_node
get_rank = utils.get_rank
get_world_size = utils.get_world_size
hostname = utils.hostname
initialize = utils.initialize
model_name = utils.model_name
set_local_rank = utils.set_local_rank
show_config = utils.show_config
spawn = utils.spawn


class UnknownBackendError(Exception):
    r"""Raised when an unknown backend is used."""


def is_main_process() -> bool:
    r"""Indicate if this process is the main process.

    By definition, the main process is the process with the global
    rank 0.

    Returns:
        ``True`` if it is the main process, otherwise ``False``.
    """
    return get_rank() == 0


def is_distributed() -> bool:
    r"""Indicate if the current process is part of a distributed group.

    Returns:
        ``True`` if the current process is part of a distributed
            group, otherwise ``False``.
    """
    return get_world_size() > 1
