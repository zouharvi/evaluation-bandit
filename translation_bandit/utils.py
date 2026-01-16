from typing import Callable
import numpy as np
import scipy.stats
import statistics
import collections
import warnings

ModelScores = dict[str, list[float]]

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Precision loss occurred"
)


def pval(scores1: list[float], scores2: list[float]) -> float:
    if len(scores1) <= 3 or len(scores2) <= 3:
        return 1.0
    val = scipy.stats.ttest_rel(
        scores1 + [np.nan] * (len(scores2) - len(scores1)),
        scores2 + [np.nan] * (len(scores1) - len(scores2)),
        nan_policy="omit",
    ).pvalue  # type: ignore
    if np.isnan(val):
        return 1.0
    return val


def pval_ind(scores1: list[float], scores2: list[float]) -> float:
    if len(scores1) <= 3 or len(scores2) <= 3:
        return 1.0
    val = scipy.stats.ttest_ind(
        scores1,
        scores2,
        nan_policy="omit",
    ).pvalue  # type: ignore
    if np.isnan(val):
        return 1.0
    return val


def tau(model_scores1: ModelScores, model_scores2: ModelScores) -> float:
    val: float = scipy.stats.kendalltau(
        [statistics.mean(model_scores1[model]) for model in model_scores1],
        [statistics.mean(model_scores2[model]) for model in model_scores1],
        variant="b",
    )[
        0
    ]  # type: ignore
    if np.isnan(val):
        return 0.0
    return val


def wtau_smooth(model_scores1: ModelScores, model_scores2: ModelScores) -> float:
    """
    weighted tau correlation. Prioritizes correct rankings for better models
    """
    val: float = scipy.stats.weightedtau(
        [statistics.mean(model_scores1[model]) for model in model_scores1],
        [statistics.mean(model_scores2[model]) for model in model_scores1],
        weigher=lambda rank: 1 / (rank + 1),
    )[
        0
    ]  # type: ignore
    if np.isnan(val):
        return 0.0
    return val


def wtau_top(model_scores1: ModelScores, model_scores2: ModelScores) -> float:
    """
    weighted tau correlation. Prioritizes correct rankings for top models
    """
    val: float = scipy.stats.weightedtau(
        [statistics.mean(model_scores1[model]) for model in model_scores1],
        [statistics.mean(model_scores2[model]) for model in model_scores1],
        weigher=lambda rank: (1 if rank <= 2 else 0.5 if rank <= 5 else 0.001),
    )[
        0
    ]  # type: ignore
    if np.isnan(val):
        return 0.0
    return val


def evalcount_smooth(
    model_scores1: ModelScores, model_scores2: ModelScores, budget: int
) -> float:
    return evalcount(
        model_scores1,
        model_scores2,
        budget,
        weigher=lambda rank: 1 / (rank + 1),
    )


def evalcount_top(
    model_scores1: ModelScores, model_scores2: ModelScores, budget: int
) -> float:
    return evalcount(
        model_scores1,
        model_scores2,
        budget,
        weigher=lambda rank: (1 if rank <= 2 else 0.5 if rank <= 5 else 0.001),
    )


def evalcount(
    model_scores1: ModelScores,
    model_scores2: ModelScores,
    budget: int,
    weigher: Callable[[int], float],
) -> float:
    # sort model_score1 by model_scores2
    model_scores1 = sorted(
        model_scores1.items(),
        key=lambda m: statistics.mean(model_scores2[m[0]]),
        reverse=True,
    )  # type: ignore
    weights = [weigher(r) for r in range(len(model_scores1))]
    weight_sum = sum(weights)

    evalcount = sum(
        [
            weights[r] / weight_sum * len(scores)
            for r, (model, scores) in enumerate(model_scores1)
        ]
    )

    model_scores2 = sorted(
        model_scores2.items(),
        key=lambda m: statistics.mean(m[1]),
        reverse=True,
    )  # type: ignore
    model_scores2_budget = {}
    for model, scores in model_scores2:
        model_scores2_budget[model] = scores[:budget]
        budget -= len(scores)
        if budget <= 0:
            break

    evalcount_maximum = sum(
        [
            weights[r] / weight_sum * len(scores)
            for r, (model, scores) in enumerate(model_scores2_budget.items())
        ]
    )

    return evalcount / evalcount_maximum


def clusters_p(model_scores: ModelScores) -> float:
    p_values = []
    # sort
    models = sorted(
        model_scores.keys(),
        key=lambda m: statistics.mean(model_scores[m]),
    )
    for model1, model2 in zip(models, models[1:]):
        p_values.append(
            pval(
                model_scores[model1],
                model_scores[model2],
            )
        )

    return statistics.mean(p_values)


def model_clusters(model_scores: ModelScores) -> float:
    clusters = 1
    # sort
    models = sorted(
        model_scores.keys(),
        key=lambda m: statistics.mean(model_scores[m]),
    )
    for model1, model2 in zip(models, models[1:]):
        if (
            pval(
                model_scores[model1],
                model_scores[model2],
            )
            < 0.05
        ):
            clusters += 1

    return clusters


def items_to_model_scores(data: list[dict], average=False) -> ModelScores:
    model_ranking = collections.defaultdict(list)
    for item in data:
        for model, score in item["scores"].items():
            model_ranking[model].append(score)
    if average:
        return {
            model: statistics.mean(scores) for model, scores in model_ranking.items()
        }
    else:
        return model_ranking


def items_to_model_ranking(data: list[dict]) -> dict[str, int]:
    model_scores = items_to_model_scores(data, average=True)
    return {
        model: rank
        for rank, model in enumerate(
            sorted(
                model_scores.keys(),
                key=lambda m: model_scores[m],
                reverse=True,
            ),
        )
    }


def load_data(
    human_scores_only=True,
    require_human_scores=True,
    wmt_years={"wmt25", "wmt24", "wmt23", "wmt23.sent"},
) -> dict[str, list[dict]]:
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt_all(normalize=False, require_human=require_human_scores)
    data = {
        f"{k[0]}_{k[1]}": [
            item
            | (
                {
                    "scores": {
                        model: model_v["human"]
                        for model, model_v in item["scores"].items()
                    }
                }
                if human_scores_only
                else {}
            )
            for item in v
        ]
        for k, v in data.items()
        if k[0] in wmt_years
    }

    return data


def load_data_bymetrics() -> dict[str, list[dict]]:
    data_all = load_data(human_scores_only=False, require_human_scores=False)
    data_out = {}
    for data_name, data in data_all.items():
        for metric in list(data[0]["scores"].values())[0].keys():
            data_new = [
                item | {
                    "scores": {
                        model: model_v[metric] if metric in model_v else None
                        for model, model_v in item["scores"].items()
                    }
                }
                for item in data
            ]
            # filter out models with more than 100 None scores
            models_to_keep = [
                model
                for model in data_new[0]["scores"].keys()
                if sum(item["scores"][model] is None for item in data_new) <= 100
            ]
            for item in data_new:
                item["scores"] = {
                    model: score
                    for model, score in item["scores"].items()
                    if model in models_to_keep
                }
            if not models_to_keep:
                continue

            # filter out lines with None scores at any model
            data_new = [
                item
                for item in data_new
                if all(
                    model_v is not None
                    for model_v in item["scores"].values()
                )
            ]
            if len(data_new) <= 100:
                continue
            data_out[f"{data_name}_{metric}"] = data_new
    return data_out

def load_data_bydomains() -> dict[str, list[dict]]:
    import subset2evaluate.utils
    import collections

    data = subset2evaluate.utils.load_data_wmt_all(normalize=False)
    data = [
        [
            item
            | {
                "scores": {
                    model: model_v["human"] for model, model_v in item["scores"].items()
                },
                "data_name": k[1],
            }
            for item in v
        ]
        for k, v in data.items()
        if k[0] == "wmt25"
    ]
    data_agg = collections.defaultdict(list)
    for data in data:
        for item in data:
            data_agg[(item["data_name"], item["domain"])].append(item)

    return {
        f"{data_name}_{domain}": items
        for (data_name, domain), items in data_agg.items()
    }


def confidence_interval(scores: list[float], confidence=0.95) -> tuple[float, float]:
    mean = statistics.mean(scores)
    sem = scipy.stats.sem(scores)
    margin = sem * scipy.stats.t.ppf((1 + confidence) / 2.0, len(scores) - 1)
    return (mean - margin, mean + margin)


def safe_mean(ys):
    return np.mean([y for y in ys if y is not None and not np.isnan(y)])
