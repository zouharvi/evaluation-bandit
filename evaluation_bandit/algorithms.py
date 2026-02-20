import math
import random
import statistics
from typing import Callable, Literal

import collections
from . import estimators
from . import utils


def uniform(data, budget) -> utils.ModelScores:
    items = data[: budget // len(data[0]["scores"])]

    return utils.items_to_model_scores(items, average=False)


def uniform_nonsquare(data, budgets: list[int]) -> list[utils.ModelScores]:
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"].keys())
    model_scores = {model: [] for model in models}
    cost = 0
    output = []

    while budgets and models:
        # randomly sample a model that's still available
        model = random.choice(models)
        # find next item for the model
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += 1

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

        # if model has no more items, remove it from available models
        if len(model_scores[model]) >= len(data):
            models.remove(model)

    return output


def successive_rejects(
    data,
    budget: int,
    phases: Literal["constant", "prioritize_all", "prioritize_top"] = "constant",
    ranking_from_elimination=False,
) -> utils.ModelScores:
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"])
    if phases == "constant":
        _phases = [
            round(2 * budget / (len(models) ** 2 + len(models) - 2))
            for i in range(len(models) - 1)
        ]
        _phases[0] = max(_phases[0], 1)
    elif phases == "prioritize_top":
        _phases = [
            round(
                budget / (len(models) ** 2 - 2 * len(models) - i * len(models) + 2 * i)
            )
            for i in range(len(models) - 1)
        ]
        _phases[0] = max(_phases[0], 1)
    elif phases == "prioritize_all":
        # phases are longer at the beginning
        # use weighting with fixed first
        _phases = [1, 0.8, 0.6]
        _phases += [0.2] * len(_phases)
        # normalize to sum to budget
        total = sum(_phases)
        _phases = [math.ceil(budget * (p / total) / (len(models) - 2)) for p in _phases]
    else:
        raise ValueError("Other (e.g. more dynamic) phase lengths not implemented yet.")

    # last phase always takes all remaining budget
    _phases[-1] = budget

    cost = 0

    model_phase_elimintation = {}
    model_scores = {model: [] for model in models}
    for phase, phase_size in enumerate(_phases):
        break_game = False
        for _ in range(phase_size):
            if not data or cost >= budget:
                break_game = True
                break
            item = data.pop(0)
            for model in models:
                cost += 1
                model_scores[model].append(item["scores"][model])
        if break_game:
            break
        # eliminate worst model
        model = min(
            models,
            key=lambda m: statistics.mean(model_scores[m]),
        )
        models.remove(model)
        model_phase_elimintation[model] = phase

    # add last models still in the game
    model_phase_elimintation |= {model: len(_phases) for model in models}

    if ranking_from_elimination:
        return model_phase_elimintation
    else:
        return {model: model_scores[model] for model in model_scores}


def successive_halving(
    data,
    budget: int,
) -> utils.ModelScores:
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"].keys())
    model_scores = {model: [] for model in models}

    rounds = int(math.log2(len(models)))

    cost = 0

    while len(models) > 1 and cost < budget and data and rounds > 0:
        budget_this_round = (budget - cost) / rounds
        # ensure we don't overspend based on items
        n_items = int(budget_this_round / len(models))

        if n_items < 1:
            break

        for _ in range(n_items):
            if not data:
                break
            item = data.pop(0)
            for model in models:
                model_scores[model].append(item["scores"][model])
                cost += 1

        rounds -= 1
        models.sort(
            key=lambda m: statistics.mean(model_scores[m])
            if model_scores[m]
            else -float("inf"),
            reverse=True,
        )
        models = models[: len(models) // 2]

    return {model: model_scores[model] for model in model_scores}


def weighted_sampling(
    data,
    budgets: list[int],
    sampling_fn: Callable[[list[float], int, int], float] = lambda scores,
    rank,
    total: 1 / (rank + 1),
    estimator_fn: estimators.Estimator = estimators.mean,
    coldstart=5,
) -> list[utils.ModelScores]:
    data = list(data)
    # cold start phase
    model_scores = {
        model: [x["scores"][model] for x in data[:coldstart]]
        for model in data[0]["scores"]
    }
    models = list(data[0]["scores"])
    cost = sum([len(model_scores[model]) for model in models])
    models.sort(key=lambda m: statistics.mean(model_scores[m]), reverse=True)

    output = []
    # active learning phase
    while len(budgets) > 0:
        model = random.choices(
            models,
            weights=[
                sampling_fn(
                    model_scores[model],
                    rank,
                    len(models),
                )
                for model, rank in zip(models, range(len(models)))
            ],
            k=1,
        )[0]
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += 1

        models = [
            model for model in model_scores if len(model_scores[model]) < len(data)
        ]
        model_estimate = estimator_fn(
            {model: scores for model, scores in model_scores.items() if model in models}
        )
        models.sort(key=model_estimate.get, reverse=True)

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

    return output


def weighted_sampling_oracle(
    data,
    budgets: list[int],
    sampling_fn: Callable[[list[float], int, int], float] = lambda scores,
    rank,
    total: 1 / (rank + 1),
    coldstart=5,
) -> list[utils.ModelScores]:
    data = list(data)
    model_scores = {model: [data[0]["scores"][model]] for model in data[0]["scores"]}
    models = list(data[0]["scores"])
    # sort according to total score
    models.sort(
        key=lambda m: statistics.mean([x["scores"][m] for x in data]),
        reverse=True,
    )
    weights = {
        model: sampling_fn(
            [x["scores"][model] for x in data],
            rank,
            len(models),
        )
        for rank, model in enumerate(models)
    }

    cost = 0
    output = []
    # allocation phase
    while len(budgets) > 0:
        model = random.choices(
            models,
            weights=[weights[m] for m in models],
            k=1,
        )[0]
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += 1

        models = [
            model for model, scores in model_scores.items() if len(scores) < len(data)
        ]

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

        if not models:
            break

    return output


def statistical_ambiguity_reduction(
    data,
    budgets: list[int],
    coldstart=5,
    weight_pointwise=1,
    weight_pairwise=1,
) -> list[utils.ModelScores]:
    data = list(data)
    models = [
        {
            "model": model,
            "ci": None,
            "pvalue": collections.defaultdict(lambda: None),
            "scores": [x["scores"][model] for x in data[:coldstart]],
        }
        for model in data[0]["scores"]
    ]
    cost = sum([len(model["scores"]) for model in models])

    def recompute_meta(model_dirty):
        nonlocal models
        model_dirty["ci"] = utils.confidence_interval(model_dirty["scores"])
        model_dirty["ci"] = model_dirty["ci"][1] - model_dirty["ci"][0]

        for model in models:
            if model == model_dirty:
                continue

            # ttest_rel might have fewer intersecting samples than ttest_ind
            # but is stronger and ultimately better for our use case
            model["pvalue"][model_dirty["model"]] = utils.pval(
                model_dirty["scores"],
                model["scores"],
            )
            model_dirty["pvalue"][model["model"]] = model["pvalue"][
                model_dirty["model"]
            ]

    for model in models:
        recompute_meta(model)

    output = []
    while budgets:
        # rank models based on ci and neighbour p-value independently
        # then average the rank
        model_rank_ci = {
            model["model"]: rank
            for rank, model in enumerate(
                sorted(
                    models,
                    key=lambda x: x["ci"],
                    reverse=True,
                )
            )
        }
        model_rank_p = {
            model["model"]: rank
            for rank, model in enumerate(
                sorted(
                    models,
                    key=lambda x: sum(x["pvalue"].values()),
                    reverse=True,
                )
            )
        }
        model = min(
            [m for m in models if len(m["scores"]) < len(data)],
            key=lambda m: weight_pointwise * model_rank_ci[m["model"]]
            + weight_pairwise * model_rank_p[m["model"]],
        )
        item = data[len(model["scores"])]
        model["scores"].append(item["scores"][model["model"]])
        cost += 1

        recompute_meta(model)

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model["model"]: list(model["scores"]) for model in models})

    return output


def upper_confidence_bound(
    data,
    budgets: list[int],
    c: float = math.sqrt(2),
    coldstart: int = 5,
    topk: int = 1,
    variant: Literal["ucb1", "lilucb"] = "ucb1",
) -> list[utils.ModelScores]:
    """
    Upper Confidence Bound (UCB1) or lil' UCB algorithm.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}
    model_counts = {model: 0 for model in models}
    model_sum_scores = {model: 0.0 for model in models}

    # To keep track of where we are in the data for each model
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            if len(model_scores[model]) < len(data):
                score = data[len(model_scores[model])]["scores"][model]
                model_scores[model].append(score)
                model_sum_scores[model] += score
                model_counts[model] += 1
                cost += 1

    while budgets and cost < budgets[0]:
        # Calculate UCB for all models
        ucb_scores = {}
        models = [model for model in models if len(model_scores[model]) < len(data)]

        total_counts = sum(model_counts.values())
        if variant == "ucb1":
            ln_total = math.log(total_counts)

        for model in models:
            mean = model_sum_scores[model] / model_counts[model]

            if variant == "ucb1":
                # UCB = mean + c * sqrt(ln(total_counts) / model_counts)
                exploration = c * math.sqrt(ln_total / model_counts[model])
            elif variant == "lilucb":
                # lil' UCB = mean + c * sqrt(ln(ln(model_counts)) / model_counts)
                # using max(..., 3) to ensure log(log(n)) > 0
                exploration = c * math.sqrt(
                    math.log(math.log(max(model_counts[model], 3)))
                    / model_counts[model]
                )
            else:
                raise ValueError(f"Unknown variant {variant}")

            ucb_scores[model] = mean + exploration

        # Select topk models
        selected_models = sorted(ucb_scores, key=ucb_scores.get, reverse=True)[:topk]

        for model in selected_models:
            score = data[len(model_scores[model])]["scores"][model]
            model_scores[model].append(score)
            model_sum_scores[model] += score
            model_counts[model] += 1
            cost += 1

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

    return output


def pvalue_rejects(
    data,
    budgets: list[int],
    threshold=0.05,
) -> list[utils.ModelScores]:
    """
    Eliminates the worst model if it is significantly worse (p < threshold)
    than the next best model. Use p_value_threshold=0.05 by default.
    """
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"].keys())
    model_scores = {model: [] for model in models}
    cost = 0
    output = []

    while budgets and len(models) > 1:
        # Round-robin sampling
        sampled_any = False
        for model in list(models):
            if not budgets:
                break
            # get next item for the model
            if len(model_scores[model]) >= len(data):
                continue
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += 1
            sampled_any = True

            while budgets and cost >= budgets[0]:
                budgets = budgets[1:]
                output.append(
                    {model: list(model_scores[model]) for model in model_scores}
                )

        if not sampled_any:
            break

        models.sort(key=lambda m: statistics.mean(model_scores[m]))

        # allow for pruning multiple models at the same time if there's difference
        while len(models) > 1:
            worst_model = models[0]
            next_worst_model = models[1]

            # Calculate p-value between worst and next worst
            p_val = utils.pval(
                model_scores[worst_model], model_scores[next_worst_model]
            )

            if p_val < threshold:
                models.remove(worst_model)
            else:
                break

    return output


def thompson_sampling(
    data,
    budgets: list[int],
    coldstart=5,
    rank_top_k=1,
) -> list[utils.ModelScores]:
    """
    Thompson Sampling for Best Arm Identification.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}
    model_counts = {model: 0 for model in models}
    model_sum_scores = {model: 0.0 for model in models}
    model_sq_sum_scores = {model: 0.0 for model in models}

    # To keep track of where we are in the data for each model
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            if len(model_scores[model]) < len(data):
                score = data[len(model_scores[model])]["scores"][model]
                model_scores[model].append(score)
                model_sum_scores[model] += score
                model_sq_sum_scores[model] += score**2
                model_counts[model] += 1
                cost += 1

    while budgets and cost < budgets[0]:
        # Generate samples for all models
        ts_samples = {}
        models = [model for model in models if len(model_scores[model]) < len(data)]

        for model in models:
            count = model_counts[model]
            mean = model_sum_scores[model] / count
            # Unbiased estimator for variance: s^2 = 1/(n-1) * sum((x_i - mean)^2)
            # sum((x_i - mean)^2) = sum(x_i^2) - 2*mean*sum(x_i) + n*mean^2
            # = sum(x_i^2) - 2*mean*(n*mean) + n*mean^2
            # = sum(x_i^2) - n*mean^2
            if count > 1:
                var = (model_sq_sum_scores[model] - count * mean**2) / (count - 1)
                # Ensure variance is non-negative and handle small variance
                var = max(var, 1e-6)
            else:
                var = 1.0  # Default variance for small samples

            # Sample from the posterior distribution of the mean
            # Standard error of the mean is sqrt(var / n)
            std_err = math.sqrt(var / count)
            sample = random.normalvariate(mean, std_err)
            ts_samples[model] = sample

        # Select topk models
        selected_models = sorted(ts_samples, key=ts_samples.get, reverse=True)[
            :rank_top_k
        ]

        for model in selected_models:
            score = data[len(model_scores[model])]["scores"][model]
            model_scores[model].append(score)
            model_sum_scores[model] += score
            model_sq_sum_scores[model] += score**2
            model_counts[model] += 1
            cost += 1

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

    return output
