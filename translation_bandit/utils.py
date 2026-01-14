import numpy as np
import scipy.stats
import statistics
import collections
import warnings

ModelScores = dict[str, list[float]]

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Precision loss occurred"
)


def pval(scores1: ModelScores, scores2: ModelScores) -> float:
    if len(scores1) < 2 or len(scores2) < 2:
        return 1.0
    return scipy.stats.ttest_rel(
        scores1 + [np.nan] * (len(scores2) - len(scores1)),
        scores2 + [np.nan] * (len(scores1) - len(scores2)),
        nan_policy="omit",
    ).pvalue


def tau(model_scores1: ModelScores, model_scores2: ModelScores) -> float:
    return scipy.stats.kendalltau(
        [statistics.mean(model_scores1[model]) for model in model_scores1],
        [statistics.mean(model_scores2[model]) for model in model_scores1],
        variant="b",
    )[0]


def wtau(model_scores1: ModelScores, model_scores2: ModelScores) -> float:
    """
    weighted tau correlation. Prioritizes correct rankings for top models
    """
    return scipy.stats.weightedtau(
        [statistics.mean(model_scores1[model]) for model in model_scores1],
        [statistics.mean(model_scores2[model]) for model in model_scores1],
        weigher=lambda rank: (
            1 if rank <= 2
            else 0.5 if rank <= 5
            else 0.001
        )
    )[0]


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

def load_data() -> dict[str, list[dict]]:
    import subset2evaluate.utils

    data = subset2evaluate.utils.load_data_wmt_all(normalize=False)
    data = {
        k[1]: [
            item
            | {
                "scores": {
                    model: model_v["human"]
                    for model, model_v in item["scores"].items()
                }
            }
            for item in v
        ]
        for k, v in data.items()
        if k[0] == "wmt25"
    }

    return data


def load_data_single(langs="en-cs_CZ") -> list[dict]:
    return load_data()[langs]


def confidence_interval(scores: list[float], confidence=0.95) -> tuple[float, float]:
    mean = statistics.mean(scores)
    sem = scipy.stats.sem(scores)
    margin = sem * scipy.stats.t.ppf((1 + confidence) / 2.0, len(scores) - 1)
    return (mean - margin, mean + margin)