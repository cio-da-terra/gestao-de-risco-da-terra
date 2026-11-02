"""Online split conformal intervals with adaptive conformal inference (ACI).

Scores are absolute residuals on the log scale from *out-of-sample*
predictions whose outcomes were already known at forecast time. ACI
(Gibbs & Candès, 2021) nudges the miscoverage level after each realised
batch: alpha <- alpha + gamma * (target - observed miss rate).
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1 - alpha) quantile.

    In theory, when alpha is too small for n scores the interval is infinite.
    ACI pushes alpha there after a run of misses; as is common in practice we
    cap at the largest observed score instead, so intervals stay usable.
    """
    n = len(scores)
    if n == 0:
        return math.inf
    level = math.ceil((n + 1) * (1 - max(alpha, 0.0))) / n
    if level > 1:
        return float(np.max(scores))
    return float(np.quantile(scores, level, method="higher"))


class ACI:
    def __init__(self, target_alpha: float, gamma: float = 0.05):
        self.target = target_alpha
        self.gamma = gamma
        self.alpha = target_alpha

    def update(self, miss_rate: float) -> None:
        self.alpha += self.gamma * (self.target - miss_rate)
        self.alpha = min(max(self.alpha, 0.0), 0.99)
