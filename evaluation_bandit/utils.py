import numpy as np
import scipy.stats
import statistics
import collections
import warnings
import itertools
import functools


ModelScores = dict[str, list[float]]
ModelEstimates = dict[str, float]

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


def tau(model_estimates1: ModelEstimates, model_estimates2: ModelEstimates) -> float:
    val: float = scipy.stats.kendalltau(
        [model_estimates1[model] for model in model_estimates1],
        [model_estimates2[model] for model in model_estimates1],
        variant="b",
    )[0]  # type: ignore
    if np.isnan(val):
        return 0.0
    return val


def stability(
    model_estimates_all: list[ModelEstimates],
    weigher=lambda rank: 1 / (rank + 1) ** 2,
) -> float:
    models = model_estimates_all[0].keys()
    model_estimates_all = [
        [model_estimates[model] for model in models]
        for model_estimates in model_estimates_all
    ]
    if len(model_estimates_all) <= 1:
        return float("nan")
    vals = [
        scipy.stats.weightedtau(a, b, weigher=weigher)[0]
        # we can't use just combinations because wtau is not symmetric
        for a, b in itertools.product(model_estimates_all, repeat=2)
    ]
    vals = [0.0 if np.isnan(x) else x for x in vals]
    return statistics.mean(vals)


def wtau(
    model_estimates1: ModelEstimates,
    model_estimates2: ModelEstimates,
    weigher=lambda rank: 1 / (rank + 1) ** 2,
) -> float:
    """
    weighted tau correlation. Prioritizes correct rankings for better models
    """
    val: float = scipy.stats.weightedtau(
        [model_estimates1[model] for model in model_estimates1],
        [model_estimates2[model] for model in model_estimates1],
        weigher=weigher,
    )[0]  # type: ignore
    if np.isnan(val):
        return 0.0
    return val


def wtau_pow(
    model_estimates1: ModelEstimates,
    model_estimates2: ModelEstimates,
    k: float,
) -> float:
    return wtau(
        model_estimates1, model_estimates2, weigher=lambda rank: 1 / (rank + 1) ** k
    )


def wtau_topk(
    model_estimates1: ModelEstimates,
    model_estimates2: ModelEstimates,
    k: int,
) -> float:
    return wtau(
        model_estimates1,
        model_estimates2,
        weigher=lambda rank: 1 if rank < k else 1 / (len(model_estimates1) - k),
    )


def wtau_botk(
    model_estimates1: ModelEstimates,
    model_estimates2: ModelEstimates,
    k: int,
) -> float:
    return wtau(
        model_estimates1,
        model_estimates2,
        weigher=lambda rank: 1
        if rank >= len(model_estimates1) - k
        else 1 / (len(model_estimates1) - k),
    )


def wtau_middlek(
    model_estimates1: ModelEstimates,
    model_estimates2: ModelEstimates,
    k: int,
) -> float:
    return wtau(
        model_estimates1,
        model_estimates2,
        weigher=lambda rank: 1
        if len(model_estimates1) / 2 - k / 2 <= rank
        and rank <= len(model_estimates1) / 2 + k / 2
        else 1 / (len(model_estimates1) - k),
    )


def wtau_revpow(
    model_estimates1: ModelEstimates, model_estimates2: ModelEstimates, k: float
) -> float:
    return wtau(
        model_estimates1,
        model_estimates2,
        weigher=lambda rank: 1 / (len(model_estimates1) - rank) ** k,
    )


def evalfocus(
    model_scores1: ModelScores,
    model_scores2: ModelScores,
    weigher=lambda rank: 1 / (rank + 1) ** 2,
) -> float:
    # sort model_score1 by model_scores2
    model_scores1 = sorted(
        model_scores1.items(),
        key=lambda m: statistics.mean(model_scores2[m[0]]),
        reverse=True,
    )
    weights = {model: weigher(r) for r, (model, _) in enumerate(model_scores1)}
    weight_sum = sum(weights.values())

    evalfocus = sum(
        [
            weights[model] / weight_sum * np.log2(max(0.5, len(scores)))
            for model, scores in model_scores1
        ]
    )

    return evalfocus**2


def avg_pval(model_scores: ModelScores) -> float:
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


def avg_payoff(model_scores: ModelScores) -> float:
    return statistics.mean([x for scores in model_scores.values() for x in scores])


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
    model_scores = collections.defaultdict(list)
    for item in data:
        for model, score in item["scores"].items():
            model_scores[model].append(score)
    if average:
        return {
            model: statistics.mean(scores) for model, scores in model_scores.items()
        }
    else:
        return model_scores


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


def model_estimates_to_model_ranking(model_estimates: ModelEstimates) -> dict[str, int]:
    return {
        model: rank
        for rank, model in enumerate(
            sorted(
                model_estimates.keys(),
                key=lambda m: model_estimates[m],
                reverse=True,
            ),
        )
    }


def data_humanscores_only(
    data: list[dict] | dict[str, list[dict]],
) -> list[dict] | dict[str, list[dict]]:
    if isinstance(data, dict):
        return {k: data_humanscores_only(v) for k, v in data.items()}
    return [
        item
        | (
            {
                "scores": {
                    model: model_v["human"] for model, model_v in item["scores"].items()
                }
            }
        )
        for item in data
    ]


def load_data(
    require_human_scores=True,
    wmt_years={"wmt25", "wmt24", "wmt23", "wmt23.sent", "wmt22", "wmt21"},
    # wmt_years={"wmt25", "wmt24"},
) -> dict[str, list[dict]]:
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt_all(
        normalize=False,
        require_human=require_human_scores,
        name_filter=lambda x: x[0] in wmt_years,
    )

    return data


def load_data_synth(
    seed=0, models=100, items=500, heteroscedastic=False, bins=None, **kwargs
) -> list[dict[str, list[dict]]]:
    import numpy as np

    random = np.random.RandomState(seed)

    models_latent = np.clip(random.normal(loc=0.70, scale=0.25, size=models), 0, 1)
    items_latent = random.normal(loc=0, scale=1, size=items)

    model_latent_mean = np.mean(models_latent)

    data_out = []
    for item_latent in items_latent:
        scores_dict = {}
        for model_i, model_latent in enumerate(models_latent):
            if heteroscedastic:
                error = random.normal(loc=0, scale=model_latent)
            else:
                error = random.normal(loc=0, scale=model_latent_mean)
            score = np.clip(model_latent + item_latent + error, 0, 1)
            if bins:
                # get closest bin, not digitize
                score = bins[np.argmin(np.abs(bins - score))]
            scores_dict[f"model_{model_i + 1}"] = {"human": float(score)}
        data_out.append({"scores": scores_dict, "cost": 1, "domain": "synth"})
    return data_out


load_data_synth_binary = functools.partial(load_data_synth, bins=[0, 1])
load_data_synth_likert = functools.partial(
    load_data_synth, bins=[0, 0.25, 0.5, 0.75, 1]
)
load_data_synth_hetero = functools.partial(load_data_synth, heteroscedastic=True)
load_data_synth_homo = functools.partial(load_data_synth, heteroscedastic=False)


def load_data_bymetrics() -> dict[str, list[dict]]:
    data_all = load_data(require_human_scores=False)
    data_out = {}
    for data_name, data in data_all.items():
        for metric in list(data[0]["scores"].values())[0].keys():
            data_new = [
                item
                | {
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
                if all(model_v is not None for model_v in item["scores"].values())
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
