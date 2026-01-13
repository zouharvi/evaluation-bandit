import statistics
from dynamic_evaluation import sampler, utils
import random
import multiprocessing
Result = dict[str, float]


def annotate_random(data, models) -> tuple[float, Result]:
    """
    Return [cost, result] for a random annotation of K models on the given data point.
    """
    item = random.choice(data)
    results = [item["scores"][model]["human"] for model in models]

    K = len(models)

    # base cost + decay per additional model
    cost = 1 + (K - 1)**0.8

    if K == 1:
        noise = random.uniform(-15, 15)
    else:
        noise = random.uniform(-3, 3)

    # TODO: use de-noiser since the linear shift is invertible
    noise_linear = 1.1
    results_mean = statistics.mean(results)
    results = {
        model: max(
            0, 
            min(100, (result - results_mean) * noise_linear + results_mean + noise)
        )
        for model, result in zip(models, results)
    }
    return cost, results

def model_clusters(model_ranking) -> float:
    clusters = 1
    # sort
    models = sorted(
        model_ranking.keys(),
        key=lambda m: statistics.mean(model_ranking[m]) if model_ranking[m] else 0
    )
    for model1, model2 in zip(models, models[1:]):
        if (
            utils.pval(
                model_ranking[model1],
                model_ranking[model2],
            )
            < 0.05
        ):
            clusters += 1

    return clusters


def _simulate(args: tuple[list[dict], sampler.Sampler]) -> tuple[list[float], list[float], list[float]]:
    """
    Returns correlation with final ranking and total cost after each match.
    """
    data, sampler = args
    costs = []
    output_clu = []
    output_cor = []

    model_ranking_true = {
        model: statistics.mean([item["scores"][model]["human"] for item in data])
        for model in sampler.models
    }

    while not costs or costs[-1] < 2_000:
        models = sampler.next_match()
        cost, results = annotate_random(data, models)
        sampler.record_match(results)
        costs.append(costs[-1] + cost if costs else cost)
        model_ranking = {
            model: sampler.model_skill(model) for model in sampler.models
        }
        output_clu.append(
            model_clusters(
                {model: sampler.scores[model] for model in sampler.models}
            )
        )
        output_cor.append(
            utils.model_correlation(model_ranking_true, model_ranking)
        )


    return costs, output_clu, output_cor

def simulate(data, sampler: sampler.Sampler, n_runs: int = 100) -> tuple[list[float], list[float], list[float]]:
    """
    Returns correlation with final ranking and total cost after each match.
    Parallelized over n_runs using multiprocessing.
    """
    with multiprocessing.Pool() as pool:
        results = pool.map(_simulate, [(data, sampler()) for _ in range(n_runs)])
    outputs_costs, outputs_clu, outputs_cor = zip(*results)
    outputs_costs = [
        statistics.mean(costs[i] for costs in outputs_costs)
        for i in range(len(outputs_costs[0]))
    ]
    outputs_clu = [
        statistics.mean(clusters[i] for clusters in outputs_clu)
        for i in range(len(outputs_clu[0]))
    ]
    outputs_cor = [
        statistics.mean(correlations[i] for correlations in outputs_cor)
        for i in range(len(outputs_cor[0]))
    ]
    return outputs_costs, outputs_clu, outputs_cor
