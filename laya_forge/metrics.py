"""Scores for one model on one labelled set: accuracy with a confidence interval, and calibration."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from laya.common import ece_score

from .model import Item


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n. Honest at small n, unlike k/n +- 2se."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(items: Sequence[Item], probs: Sequence[np.ndarray]) -> dict[str, Any]:
    """Accuracy, ECE, Brier and NLL, overall and per question."""
    by_q: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_q[it.qid].append(i)

    def block(idx: list[int]) -> dict[str, Any]:
        conf = np.array([probs[i].max() for i in idx])
        correct = np.array([int(probs[i].argmax() == items[i].label) for i in idx])
        onehot = [np.eye(items[i].k)[items[i].label] for i in idx]
        k = int(correct.sum())
        lo, hi = wilson(k, len(idx))
        return {
            "n": len(idx),
            "accuracy": k / len(idx),
            "accuracy_ci": [round(lo, 4), round(hi, 4)],
            "ece": ece_score(conf, correct, bins=10),
            "brier": float(np.mean([((probs[i] - y) ** 2).sum() for i, y in zip(idx, onehot, strict=True)])),
            "nll": float(-np.mean([np.log(max(probs[i][items[i].label], 1e-12)) for i in idx])),
        }

    return {
        "overall": block(list(range(len(items)))),
        "questions": {q: block(idx) for q, idx in sorted(by_q.items())},
    }
