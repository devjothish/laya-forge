"""Temperature scaling, fitted per Laya temperature bucket on held-out labels.

Laya divides logits by one temperature per (question type, option count) bucket, so that is the
granularity a checkpoint can carry. The search is confined to the range Laya's loader accepts
([0.5, 5]); a value outside it would be clamped at load time and the fitted calibration lost.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from laya.common import TEMP_MAX, TEMP_MIN

from .model import Item

GRID = np.round(np.arange(TEMP_MIN, TEMP_MAX + 1e-9, 0.01), 2)


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64) / t
    e = np.exp(z - z.max())
    return np.asarray(e / e.sum())


def nll(logits: Sequence[np.ndarray], labels: Sequence[int], t: float) -> float:
    return float(
        -np.mean([np.log(max(softmax(z, t)[y], 1e-12)) for z, y in zip(logits, labels, strict=True)])
    )


def fit_temperatures(logits: Sequence[np.ndarray], items: Sequence[Item]) -> dict[str, float]:
    """Temperature per bucket that minimises negative log-likelihood of the true answers."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        groups[it.bucket].append(i)
    fitted = {}
    for bucket, idx in groups.items():
        zs, ys = [logits[i] for i in idx], [items[i].label for i in idx]
        fitted[bucket] = float(min(GRID, key=lambda t: nll(zs, ys, t)))
    return fitted


def probabilities(
    logits: Sequence[np.ndarray], items: Sequence[Item], temps: dict[str, float]
) -> list[np.ndarray]:
    return [softmax(z, temps.get(it.bucket, 1.0)) for z, it in zip(logits, items, strict=True)]
