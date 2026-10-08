# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
# Modified by Morrowmake for CMP 170HX support; see repository history.
import contextlib
import functools
from typing import Any, Callable
import torch

def _tensor_snapshot(x: Any) -> tuple | None:
    """What a cached result depends on besides the identity of a tensor
    argument: storage address, view geometry and version counter (bumped by
    every in-place write). Inference tensors keep no version counter; they get
    ``None`` there, and their address and geometry still have to match."""
    if not isinstance(x, torch.Tensor):
        return None
    try:
        version = x._version
    except RuntimeError:
        version = None
    try:
        ptr = x.data_ptr()
    except RuntimeError:
        ptr = None
    return (ptr, x.storage_offset(), tuple(x.shape), tuple(x.stride()), version)


def tensor_cache(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """
    A decorator that caches the most recent results of a function with tensor inputs.

    This decorator will store the output of the decorated function for the most recent set of input tensors.
    The cache is limited to a fixed size (default is 8). When the cache is full, the oldest entry will be removed.

    An entry is reused only for the very same argument objects whose contents
    have not been written since the entry was stored: each tensor argument's
    storage address, view geometry and version counter are recorded at store
    time and compared on lookup, so a buffer rewritten in place (for example a
    persistent ``query_start_loc`` reused across steps) is recomputed instead
    of returning the result for its old contents.

    Args:
        fn (Callable[..., torch.Tensor]):
            The function to be decorated. It should take tensor inputs and return tensor outputs.

    Returns:
        Callable[..., torch.Tensor]:
            A wrapped version of the input function with single-entry caching.
    """

    cache_entries: list[tuple[tuple, dict, tuple, dict, Any]] = []
    cache_size = 8
    cached_context = None

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries, cache_size, cached_context
        # Metadata is immutable within one model forward, but buffers may be
        # reused in inference mode across forwards without version counters.
        try:
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
        except (AssertionError, RuntimeError):
            context = None
        if context is not cached_context:
            cache_entries = []
            cached_context = context
        if context is None and any(
            isinstance(x, torch.Tensor) and _tensor_snapshot(x)[-1] is None
            for x in (*args, *kwargs.values())
        ):
            return fn(*args, **kwargs)
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_snaps, last_kwsnaps, last_result = entry
            if (
                len(args) == len(last_args)
                and len(kwargs) == len(last_kwargs)
                and all(a is b for a, b in zip(args, last_args))
                and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                )
                and all(_tensor_snapshot(a) == s for a, s in zip(args, last_snaps))
                and all(
                    _tensor_snapshot(v) == last_kwsnaps[k] for k, v in kwargs.items()
                )
            ):
                cache_entries = cache_entries[:i] + cache_entries[i + 1 :] + [entry]
                return last_result

        result = fn(*args, **kwargs)

        # Snapshot after the call: the decorated functions only read their inputs.
        snaps = tuple(_tensor_snapshot(a) for a in args)
        kwsnaps = {k: _tensor_snapshot(v) for k, v in kwargs.items()}
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, snaps, kwsnaps, result))
        return result

    return wrapper

