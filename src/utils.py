import scipy.stats
import statistics
import collections
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Precision loss occurred"
)


def pval(scores1, scores2) -> float:
    if len(scores1) < 2 or len(scores2) < 2:
        return 1.0
    return scipy.stats.ttest_ind(
        scores1,
        scores2,
        equal_var=False,
        alternative="less",
    ).pvalue


def tau(model_ranking1, model_ranking2) -> float:
    return scipy.stats.kendalltau(
        [model_ranking1[model] for model in model_ranking1],
        [model_ranking2[model] for model in model_ranking1],
        variant="b",
    )[0]


def wtau(model_ranking1, model_ranking2) -> float:
    """
    weighted tau correlation. Prioritizes correct rankings for top models
    """
    return scipy.stats.weightedtau(
        [model_ranking1[model] for model in model_ranking1],
        [model_ranking2[model] for model in model_ranking1],
        # weigher=lambda x: (
        #     1 if x < 3
        #     else 0.5 if x < 5
        #     else 0.1
        # )
    )[0]


def model_clusters(model_ranking) -> float:
    clusters = 1
    # sort
    models = sorted(
        model_ranking.keys(),
        key=lambda m: statistics.mean(model_ranking[m]) if model_ranking[m] else 0,
    )
    for model1, model2 in zip(models, models[1:]):
        if (
            pval(
                model_ranking[model1],
                model_ranking[model2],
            )
            < 0.05
        ):
            clusters += 1

    return clusters


def items_to_model_scores(data, average=False):
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


def items_to_model_ranking(data):
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

def load_data():
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


def load_data_single(langs="en-ko_KR"):
    return load_data()[langs]