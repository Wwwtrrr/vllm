# SPDX-License-Identifier: Apache-2.0
"""Stream CPU safetensors IO comparisons without loading either full model."""

import json
from pathlib import Path

import torch
from safetensors import safe_open

ALIASES = {
    "embedding": (
        "embed.weight",
        "model.embed.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    ),
    "head": (
        "head.weight",
        "model.head.weight",
        "model.lm_head.weight",
        "lm_head.weight",
    ),
}


def locate_io(directory, kind):
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Local checkpoint directory does not exist: {root}")
    index = root / "model.safetensors.index.json"
    if index.exists():
        mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    else:
        mapping = {}
        for file in root.glob("*.safetensors"):
            with safe_open(str(file), framework="pt", device="cpu") as stream:
                for key in stream.keys():  # noqa: SIM118 (safe_open is not iterable)
                    if key in ALIASES[kind]:
                        if key in mapping:
                            raise ValueError(f"Duplicate IO tensor {key}")
                        mapping[key] = file.name
    found = [key for key in ALIASES[kind] if key in mapping]
    if len(found) > 1:
        raise ValueError(f"Ambiguous {kind} IO aliases: {found}")
    return (root / mapping[found[0]], found[0]) if found else None


def compare_io(left_directory, right_directory, chunk_rows=128):
    """Compare after BF16 conversion, matching this reference's compute dtype."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    report = {}
    for kind in ALIASES:
        left, right = locate_io(left_directory, kind), locate_io(right_directory, kind)
        if left is None or right is None:
            report[kind] = dict(
                status="missing",
                left_present=left is not None,
                right_present=right is not None,
            )
            continue
        with (
            safe_open(str(left[0]), framework="pt", device="cpu") as a,
            safe_open(str(right[0]), framework="pt", device="cpu") as b,
        ):
            x, y = a.get_slice(left[1]), b.get_slice(right[1])
            shape = x.get_shape()
            if x.get_dtype() not in ("BF16", "F16", "F32") or y.get_dtype() not in (
                "BF16",
                "F16",
                "F32",
            ):
                report[kind] = dict(
                    status="unsupported_dtype",
                    left_dtype=x.get_dtype(),
                    right_dtype=y.get_dtype(),
                )
                continue
            if len(shape) != 2 or shape != y.get_shape():
                report[kind] = dict(
                    status="shape_mismatch", left_shape=shape, right_shape=y.get_shape()
                )
                continue
            total, unequal, absolute_sum, maximum, nonfinite = 0, 0, 0.0, 0.0, 0
            for start in range(0, shape[0], chunk_rows):
                xx = x[start : start + chunk_rows].to(torch.bfloat16).float()
                yy = y[start : start + chunk_rows].to(torch.bfloat16).float()
                finite = torch.isfinite(xx) & torch.isfinite(yy)
                nonfinite += int((~finite).sum())
                delta = (xx - yy).abs()[finite]
                total += xx.numel()
                unequal += int((xx != yy).sum())
                absolute_sum += float(delta.double().sum())
                if delta.numel():
                    maximum = max(maximum, float(delta.max()))
            report[kind] = dict(
                status="compared",
                shape=shape,
                compared_dtype="bfloat16",
                unequal=unequal,
                total=total,
                nonfinite=nonfinite,
                max_abs=maximum if not nonfinite else None,
                mean_abs=absolute_sum / total if total and not nonfinite else None,
                equal=unequal == 0 and nonfinite == 0,
            )
    return report
