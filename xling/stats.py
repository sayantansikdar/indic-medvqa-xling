"""Metrics and paired statistics (Section 6.8)."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, stat=np.mean, alpha: float = 0.05):
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    boots = np.array([stat(v[i]) for i in idx])
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_ratio_ci(num, den, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """CI for sum(num)/sum(den), resampling items. num, den: (N,) or (N, K)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    point = num.sum(0) / den.sum(0)
    boots = np.stack([num[i].sum(0) / den[i].sum(0) for i in idx])
    return point, np.quantile(boots, alpha / 2, axis=0), np.quantile(boots, 1 - alpha / 2, axis=0)


def mcnemar(correct_a, correct_b) -> float:
    """Exact McNemar p-value for paired correctness vectors."""
    from scipy.stats import binomtest

    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def ece(confidence, correct, n_bins: int = 10) -> float:
    c, y = np.asarray(confidence, float), np.asarray(correct, float)
    bins = np.minimum((c * n_bins).astype(int), n_bins - 1)
    return float(sum(abs(c[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(n_bins) if (bins == b).any()))


def gap_recovery_ratio(acc_after: float, acc_before: float, acc_en: float) -> float:
    gap = acc_en - acc_before
    return float("nan") if gap <= 0 else (acc_after - acc_before) / gap


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al., 2019) between (N, d) representation matrices."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(num / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")))
