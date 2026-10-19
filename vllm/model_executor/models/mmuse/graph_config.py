# SPDX-License-Identifier: Apache-2.0
"""Explicit FULL graph configuration for Qwen3/MMuse.

Raw capture sizes are shared by the target and draft graph managers. Include
both query widths so rounding does not drop a maximum-batch draft descriptor.
Compilation and graph replay are separate switches: mode=0 retains native
operators while allowing outer FULL graph capture; mode=3 opts into compilation.
"""


def build_mmuse_full_graph_config(
    block_size: int, max_num_seqs: int, *, compilation_mode: int = 0
) -> dict:
    if block_size <= 0 or max_num_seqs <= 0:
        raise ValueError("block_size and max_num_seqs must be positive")
    if compilation_mode not in (0, 3):
        raise ValueError("MMuse FULL configuration supports compilation_mode 0 or 3")
    # Also cover ordinary one-token decode after speculative batches shrink.
    sizes = set(range(1, max_num_seqs + 1))
    for count in range(1, max_num_seqs + 1):
        sizes.add(count * block_size)
        sizes.add(count * (block_size + 1))
    return {
        "mode": compilation_mode,
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": sorted(sizes),
        "max_cudagraph_capture_size": max(sizes),
    }
