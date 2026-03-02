import math
import random
import statistics
from typing import Callable, Literal

import collections
from . import estimators
from . import utils


def uniform(data, budgets: list[int]) -> list[utils.ModelScores]:
    items = []
    cost = 0
    models_count = len(data[0]["scores"])
    data = list(data)
    output = []
    while budgets and data:
        budget = budgets[0]
        item = data.pop(0)
        cost += item["cost"] * models_count
        if cost >= budget:
            budgets = budgets[1:]
            output.append(utils.items_to_model_scores(items, average=False))
        items.append(item)

    return output


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
        cost += item["cost"]

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

    model_scores = {model: [] for model in models}
    for phase, phase_size in enumerate(_phases):
        break_game = False
        for _ in range(phase_size):
            if not data or cost >= budget:
                break_game = True
                break
            item = data.pop(0)
            for model in models:
                cost += item["cost"]
                model_scores[model].append(item["scores"][model])
        if break_game:
            break
        # eliminate worst model
        model = min(
            models,
            key=lambda m: statistics.mean(model_scores[m]),
        )
        models.remove(model)

    return model_scores


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
                cost += item["cost"]

        rounds -= 1
        models.sort(
            key=lambda m: statistics.mean(model_scores[m])
            if model_scores[m]
            else -float("inf"),
            reverse=True,
        )
        models = models[: len(models) // 2]

    return model_scores


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
    cost = sum(x["cost"] for x in data[:coldstart]) * len(models)

    output = []
    # active learning phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        # we want to estimate given the context of all models, not just the ones running
        model_estimate = estimator_fn(model_scores)
        models.sort(key=model_estimate.get, reverse=True)

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
        cost += item["cost"]

        models = [
            model for model in model_scores if len(model_scores[model]) < len(data)
        ]

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

    cost = data[0]["cost"] * len(models)
    output = []
    # allocation phase
    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        model = random.choices(
            models,
            weights=[weights[m] for m in models],
            k=1,
        )[0]
        item = data[len(model_scores[model])]
        model_scores[model].append(item["scores"][model])
        cost += item["cost"]

        models = [
            model for model, scores in model_scores.items() if len(scores) < len(data)
        ]

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
    cost = sum(x["cost"] for x in data[:coldstart]) * len(models)

    def recompute_meta(model_dirty):
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
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model["model"]: list(model["scores"]) for model in models})
            continue

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
        cost += item["cost"]

        recompute_meta(model)

    return output


def upper_confidence_bound(
    data,
    budgets: list[int],
    c: float = 100 * math.sqrt(2),
    coldstart: int = 5,
    topk: int = 1,
    variant: Literal["ucb1", "lilucb"] = "ucb1",
    estimator_fn: Callable[[list[float]], float] = estimators.mean,
) -> list[utils.ModelScores]:
    """
    Upper Confidence Bound (UCB1) or lil' UCB algorithm.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}

    # To keep track of where we are in the data for each model
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        # Calculate UCB for all models

        if variant == "ucb1":
            ln_total = math.log(sum(len(model_scores[model]) for model in models))

        ucb_scores = {}
        # we want to estimate given the context of all models, not just the ones running
        model_estimates = estimator_fn(model_scores)
        for model in models:
            if variant == "ucb1":
                # UCB = mean + c * sqrt(ln(total_counts) / model_counts)
                exploration = c * math.sqrt(ln_total / len(model_scores[model]))
            elif variant == "lilucb":
                # lil' UCB = mean + c * sqrt(ln(ln(model_counts)) / model_counts)
                # using max(..., 3) to ensure log(log(n)) > 0
                exploration = c * math.sqrt(
                    math.log(math.log(max(len(model_scores[model]), 3)))
                    / len(model_scores[model])
                )
            else:
                raise ValueError(f"Unknown variant {variant}")

            ucb_scores[model] = model_estimates[model] + exploration

        # Select topk models
        selected_models = sorted(ucb_scores, key=ucb_scores.get, reverse=True)[:topk]

        for model in selected_models:
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

        models = [model for model in models if len(model_scores[model]) < len(data)]

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

    while budgets:
        if cost >= budgets[0] or all(
            len(model_scores[model]) >= len(data) for model in models
        ):
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        # Round-robin sampling
        for model in list(models):
            # get next item for the model
            if len(model_scores[model]) >= len(data):
                continue
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

        models.sort(key=lambda m: statistics.mean(model_scores[m]))

        # allow for pruning multiple models at the same time if there's a statistical difference
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
    estimator_fn: Callable[[list[float]], float] = estimators.mean,
) -> list[utils.ModelScores]:
    """
    Thompson Sampling for Best Arm Identification.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}

    # To keep track of where we are in the data for each model
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        # Generate samples for all models
        ts_samples = {}
        models = [model for model in models if len(model_scores[model]) < len(data)]

        # we want to estimate given the context of all models, not just the ones running
        model_estimates = estimator_fn(model_scores)
        for model in models:
            sample = random.normalvariate(
                model_estimates[model], statistics.stdev(model_scores[model])
            )
            ts_samples[model] = sample

        # Select topk models
        selected_models = sorted(ts_samples, key=ts_samples.get, reverse=True)[
            :rank_top_k
        ]

        for model in selected_models:
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

    return output


def greedy_oracle(
    data,
    budgets: list[int],
    coldstart=5,
    batch_size=10,
    estimator_fn: Callable[[list[float]], float] = estimators.mean,
) -> list[utils.ModelScores]:
    """
    Greedy Oracle for Best Arm Identification. This version massively overfits to variance.
    """
    # Initialize data and models
    data = list(data)
    models = list(data[0]["scores"].keys())

    # Track scores and counts for each model
    model_scores = {model: [] for model in models}
    model_estimates_true = estimator_fn(
        utils.items_to_model_scores(data, average=False)
    )

    # To keep track of where we are in the data for each model
    cost = 0
    output = []

    # Cold start: sample each model coldstart times
    for _ in range(coldstart):
        for model in models:
            item = data[len(model_scores[model])]
            model_scores[model].append(item["scores"][model])
            cost += item["cost"]

    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append({model: list(model_scores[model]) for model in model_scores})
            continue

        lookahead_wtau = []
        for model in models:
            # extend by batch size and compute wtau
            model_scores_local = {
                _model: scores
                + (
                    [
                        x["scores"][_model]
                        for x in data[len(scores) : len(scores) + batch_size]
                    ]
                    if model == _model
                    else []
                )
                for _model, scores in model_scores.items()
            }
            wtau = utils.wtau(
                estimator_fn(model_scores_local),
                model_estimates_true,
            )
            lookahead_wtau.append((wtau, model))

        # take the model with the highest wtau
        model_best = max(lookahead_wtau, key=lambda x: x[0])[1]
        scores_to_add = data[
            len(model_scores[model_best]) : len(model_scores[model_best]) + batch_size
        ]
        model_scores[model_best] += [x["scores"][model_best] for x in scores_to_add]
        cost += sum([x["cost"] for x in scores_to_add])

        models = [model for model in models if len(model_scores[model]) < len(data)]

    return output


def greedy_oracle_invariant(
    data,
    budgets: list[int],
    coldstart=5,
    batch_size=10,
    shuffle_repetitions=3,
    estimator_fn: Callable[[list[float]], float] = estimators.mean,
) -> list[utils.ModelScores]:
    """
    Greedy Oracle for Best Arm Identification invariant to data ordering.
    """
    # Initialize data and models
    data = list(data)
    data_orig = list(data)
    models = list(data[0]["scores"].keys())

    if not all([x["cost"] == 1 for x in data]):
        raise ValueError("All data must have cost 1")

    # Track scores and counts for each model
    model_mask = {model: coldstart for model in models}
    model_estimates_true = estimator_fn(
        utils.items_to_model_scores(data, average=False)
    )

    def mask_to_model_scores(mask, data):
        return {
            model: [x["scores"][model] for x in data[: mask[model]]] for model in mask
        }

    # To keep track of where we are in the data for each model
    cost = coldstart * len(models)
    output = []

    while budgets:
        if cost >= budgets[0] or not models:
            budgets = budgets[1:]
            output.append(mask_to_model_scores(model_mask, data_orig))
            continue

        lookahead_wtau = collections.defaultdict(list)
        # create stochasticity to not overfit to data ordering
        # shuffle multiple times
        for _ in range(shuffle_repetitions):
            random.shuffle(data)
            for model in models:
                # extend by batch size and compute wtau
                model_mask_local = model_mask | {model: model_mask[model] + batch_size}
                wtau = utils.wtau(
                    estimator_fn(mask_to_model_scores(model_mask_local, data)),
                    model_estimates_true,
                )
                lookahead_wtau[model].append(wtau)

        # take the model with the highest wtau
        lookahead_wtau = {
            model: statistics.mean(wtaus) for model, wtaus in lookahead_wtau.items()
        }
        model_best = max(lookahead_wtau, key=lookahead_wtau.get)
        model_mask[model_best] += batch_size
        cost += batch_size

        models = [model for model in models if model_mask[model] < len(data_orig)]

    return output
