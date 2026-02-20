import statistics
from typing import Callable
import numpy as np

Estimator = Callable[dict[str, list[float]], dict[str, float]]


def mean(model_scores: dict[str, list[float]]) -> dict[str, float]:
    return {model: statistics.mean(scores) for model, scores in model_scores.items()}


def additive(model_scores: dict[str, list[float]], iters: int = 10) -> dict[str, float]:
    models = list(model_scores.keys())
    len_max = max(len(scores) for scores in model_scores.values())
    # Convert to float array; missing items should be np.nan
    X = np.array(
        [model_scores[m] + [np.nan] * (len_max - len(model_scores[m])) for m in models],
        dtype=float,
    )

    mask = ~np.isnan(X)
    mu = X[mask].mean() if mask.any() else 0.0

    b_m = np.zeros(X.shape[0])
    b_d = np.zeros(X.shape[1])

    for _ in range(iters):
        # Update b_d (items)
        res_d = X - mu - b_m[:, None]
        b_d = np.divide(
            np.nansum(res_d * mask, axis=0),
            mask.sum(axis=0),
            out=b_d,
            where=mask.sum(axis=0) > 0,
        )

        # Update b_m (models)
        res_m = X - mu - b_d[None, :]
        b_m = np.divide(
            np.nansum(res_m * mask, axis=1),
            mask.sum(axis=1),
            out=b_m,
            where=mask.sum(axis=1) > 0,
        )

    return {models[i]: mu + b_m[i] for i in range(len(models))}
