import statistics
from DUKE import sampler
Result = sampler.Result
import random

def annotate_random(data, models) -> tuple[float, Result]:
    """
    Return [cost, result] for a random annotation of K models on the given data point.
    """
    item = random.choice(data)
    results = [
        item["scores"][model]["human"]
        for model in models
    ]

    K = len(models)

    # base cost + per additional model
    cost = 1 + 0.7 * (K - 1)

    if K == 1:
        noise = random.uniform(-15, 15)
    else:
        noise = random.uniform(-3, 3)

    # TODO: use de-noiser since the linear shift is inversible
    noise_linear = 1.1
    results_mean = statistics.mean(results)
    results = {
        model: max(0, min(100, (result - results_mean) * noise_linear + results_mean + noise))
        for model, result in zip(models, results)
    }
    return cost, results


def model_correlation(model_ranking1, model_ranking2) -> float:
    return statistics.correlation(
        [model_ranking1[model] for model in model_ranking1],
        [model_ranking2[model] for model in model_ranking1],
        method="ranked",
    )


def simulate(data, sampler: sampler.Sampler) -> list[tuple[float, float]]:
    """
    Returns correlation with final ranking and total cost after each match.
    """

    sampler = sampler()
    costs_all = []
    correlations_all = []

    for _ in range(5):
        costs = []
        correlations = []

        model_ranking_true = {
            model: statistics.mean([
                item["scores"][model]["human"]
                for item in data
            ])
            for model in sampler.models
        }

        while not costs or costs[-1] < 1_000:
            models = sampler.next_match()
            cost, results = annotate_random(data, models)
            sampler.record_match(results)
            costs.append(costs[-1] + cost if costs else cost)
            model_ranking = {
                model: sampler.model_skill(model)
                for model in sampler.models
            }
            correlations.append(
                model_correlation(model_ranking_true, model_ranking)
            )
        costs_all.append(costs)
        correlations_all.append(correlations)

    costs_all = [statistics.mean(costs[i] for costs in costs_all) for i in range(len(costs_all[0]))]
    correlations_all = [statistics.mean(correlations[i] for correlations in correlations_all) for i in range(len(correlations_all[0]))]
    return list(zip(costs_all, correlations_all))
