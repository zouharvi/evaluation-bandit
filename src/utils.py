import scipy.stats
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred")

def pval(scores1, scores2) -> float:
    if len(scores1) < 2 or len(scores2) < 2:
        return 1.0
    return scipy.stats.ttest_ind(
        scores1, scores2,
        equal_var=False,
        alternative="less",
    ).pvalue


def model_correlation(model_ranking1, model_ranking2) -> float:
    return scipy.stats.kendalltau(
        [model_ranking1[model] for model in model_ranking1],
        [model_ranking2[model] for model in model_ranking1],
        variant="b",
    )

def model_correlation_weighted(model_ranking1, model_ranking2, weights) -> float:
    return scipy.stats.weightedtau(
        [model_ranking1[model] for model in model_ranking1],
        [model_ranking2[model] for model in model_ranking1],
        weights=[weights[model] for model in model_ranking1],
        variant="b",
    )