import math
import random
from typing import Any, Callable, Literal

import collections
from . import utils


def uniform(data, budget) -> utils.ModelScores:
    items = random.sample(data, k=budget // len(data[0]["scores"]))

    return utils.items_to_model_scores(items, average=False)


def uniform_random(data, budget) -> utils.ModelScores:
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"].keys())
    model_scores = {model: [] for model in models}
    cost = 0
    while cost < budget and models:
        # randomly sample a model that's still available
        model = random.choice(models)
        # find next item for the model
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += 1
        # if model has no more items, remove it from available models
        if len(model_scores[model]) >= len(data):
            models.remove(model)

    return model_scores


def successive_rejects(
    data,
    budget,
    phases: Literal["constant", "prioritize_all", "prioritize_top"] = "constant",
    ranking_from_elimination=False,
) -> utils.ModelScores:
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"])
    if phases == "constant":
        _phases: Any = [
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

    # expected_cost = sum(
    #     [m * phase for m, phase in zip(range(len(models), 1, -1), _phases)]
    # )
    # if expected_cost > budget * 1.25 or expected_cost < budget * 0.75:
    #     print("Warning: budget too small/large for the selected phases. Expected cost:", expected_cost, "Budget:", budget)

    # last phase always takes all remaining budget
    _phases[-1] = budget

    # last phase needs to have at least two models still

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
            key=lambda m: utils.statistics.mean(model_scores[m]),
        )
        models.remove(model)
        model_phase_elimintation[model] = phase

    # add last models still in the game
    model_phase_elimintation |= {model: len(_phases) for model in models}

    if ranking_from_elimination:
        return model_phase_elimintation
    else:
        return {model: model_scores[model] for model in model_scores}


def weighted_sampling(
    data,
    budgets: list[int],
    sampling_fn: Callable[[list[float], int, int], float] = lambda x, rank, total: 1
    / (rank + 1),
    coldstart=5,
) -> list[utils.ModelScores]:
    """
    Rank-based epsilon-greedy approach
    """
    data = list(data)
    model_index = {model: 0 for model in data[0]["scores"]}
    model_scores = {model: [] for model in data[0]["scores"]}
    models = list(data[0]["scores"])
    cost = 0
    # cold start phase
    for _ in range(coldstart):
        for model in models:
            item = data[model_index[model]]
            model_scores[model].append(item["scores"][model])
            model_index[model] += 1
            cost += 1
    models.sort(key=lambda m: utils.statistics.mean(model_scores[m]), reverse=True)

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
        item = data[model_index[model]]
        model_scores[model].append(item["scores"][model])
        model_index[model] += 1
        cost += 1

        models = [model for model, i in model_index.items() if i < len(data)]
        models.sort(key=lambda m: utils.statistics.mean(model_scores[m]), reverse=True)

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

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
            "index": 0,
            "ci": None,
            "pvalue": collections.defaultdict(lambda: None),
            "scores": [],
        }
        for model in data[0]["scores"]
    ]
    cost = 0
    # cold start phase
    for _ in range(coldstart):
        for model in models:
            item = data[model["index"]]
            model["scores"].append(item["scores"][model["model"]])
            model["index"] += 1
            cost += 1

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
            [m for m in models if m["index"] < len(data)],
            key=lambda m: weight_pointwise * model_rank_ci[m["model"]]
            + weight_pairwise * model_rank_p[m["model"]],
        )
        item = data[model["index"]]
        model["scores"].append(item["scores"][model["model"]])
        model["index"] += 1
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
) -> list[utils.ModelScores]:
    """
    Upper Confidence Bound (UCB1) algorithm.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}
    model_counts = {model: 0 for model in models}
    model_sum_scores = {model: 0.0 for model in models}

    # To keep track of where we are in the data for each model
    model_index = {model: 0 for model in models}
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            if model_index[model] < len(data):
                score = data[model_index[model]]["scores"][model]
                model_scores[model].append(score)
                model_sum_scores[model] += score
                model_counts[model] += 1
                model_index[model] += 1
                cost += 1

    while budgets and cost < budgets[0]:
        # Calculate UCB for all models
        # UCB = mean + c * sqrt(ln(total_counts) / model_counts)
        ucb_scores = {}
        models = [model for model in models if model_index[model] < len(data)]

        total_counts = sum(model_counts.values())
        ln_total = math.log(total_counts)
        for model in models:
            mean = model_sum_scores[model] / model_counts[model]
            exploration = c * math.sqrt(ln_total / model_counts[model])
            ucb_scores[model] = mean + exploration

        # Select topk models
        selected_models = sorted(ucb_scores, key=ucb_scores.get, reverse=True)[:topk]

        for model in selected_models:
            score = data[model_index[model]]["scores"][model]
            model_scores[model].append(score)
            model_sum_scores[model] += score
            model_counts[model] += 1
            model_index[model] += 1
            cost += 1

        if cost >= budgets[0]:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})

    return output
