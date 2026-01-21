import collections
import math
from typing import Callable
from evaluation_bandit import utils
import numpy as np
import concurrent.futures

BUDGETS = np.linspace(0.1, 0.9, 20, dtype=float)


def _simulate(args):
    (
        seed,
        data_name,
        data,
        fn,
        fn_kwargs,
        accepts_budgets,
        ranking_only,
        BUDGETS,
    ) = args
    budgets = [int(len(data) * len(data[0]["scores"]) * b) for b in BUDGETS]
    if accepts_budgets:
        model_scores_all = fn(data, budgets=budgets, **fn_kwargs)
    else:
        model_scores_all = [fn(data, budget, **fn_kwargs) for budget in budgets]
    model_scores_true = {
        model: [item["scores"][model] for item in data] for model in data[0]["scores"]
    }
    output = []
    for budget_p, budget, model_scores in zip(BUDGETS, budgets, model_scores_all):
        if not ranking_only:
            output.append(
                {
                    "budget": budget_p,
                    "tau": utils.tau(model_scores, model_scores_true),
                    "wtau_smooth": utils.wtau_smooth(model_scores, model_scores_true),
                    "wtau_top": utils.wtau_top(model_scores, model_scores_true),
                    "clup": utils.clusters_p(model_scores),
                    "evalcount_smooth": utils.evalcount_smooth(
                        model_scores, model_scores_true, budget
                    ),
                    "evalcount_top": utils.evalcount_top(
                        model_scores, model_scores_true, budget
                    ),
                }
            )
        else:
            raise NotImplementedError
    return [result | {"data_name": data_name, "seed": seed} for result in output]


def simulate(
    fn: Callable,
    data_names=None,
    seeds=1,
    fn_kwargs={},
    accepts_budgets=False,
    ranking_only=False,
    fn_data_all=utils.load_data,
):
    print("Running", fn.__name__, "with", fn_kwargs)

    data_all = [
        (
            seed,
            data_name,
            data,
            fn,
            fn_kwargs,
            accepts_budgets,
            ranking_only,
            BUDGETS,
        )
        for data_name, data in fn_data_all().items()
        if data_names is None or data_name in data_names
        for seed in range(seeds)
    ]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        output = [item for list in executor.map(_simulate, data_all) for item in list]

    # aggregate across seeds
    data_agg = collections.defaultdict(list)
    for item in output:
        data_agg[(item["data_name"], item["budget"])].append(item)

    def compute_stats(xs):
        keys = [
            "tau",
            "wtau_smooth",
            "wtau_top",
            "clup",
            "evalcount_smooth",
            "evalcount_top",
        ]
        out = {
            "data_name": xs[0]["data_name"],
            "budget": xs[0]["budget"],
        }
        for key in keys:
            out[key] = np.mean([x[key] for x in xs])
            out[key + "_ci"] = utils.confidence_interval([x[key] for x in xs])
        return out

    return [compute_stats(cs) for cs in data_agg.values()]


def _simulate_subset2evaluate(args):
    (
        data_name,
        data,
        fn_kwargs,
        BUDGETS,
    ) = args
    import subset2evaluate.select_subset

    budgets = [int(len(data) * len(data[0]["scores"]) * b) for b in BUDGETS]
    # hotfix for missing values
    for item in data:
        for model in item["scores"]:
            item["scores"][model]["MetricX-25"] = item["scores"][model].get(
                "MetricX-25", 0
            )
    data_s2e = subset2evaluate.select_subset.basic(data, **fn_kwargs)
    model_scores_true = {
        model: [item["scores"][model]["human"] for item in data]
        for model in data[0]["scores"]
    }
    output = []
    for budget_p, budget in zip(BUDGETS, budgets):
        cost = 0
        model_scores = collections.defaultdict(list)
        for item in data_s2e:
            for model, model_score in item["scores"].items():  # type: ignore
                model_scores[model].append(model_score["human"])
                cost += 1
                if cost > budget:
                    break
            if cost > budget:
                break
        output.append(
            {
                "budget": budget_p,
                "tau": utils.tau(model_scores, model_scores_true),
                "wtau_smooth": utils.wtau_smooth(model_scores, model_scores_true),
                "wtau_top": utils.wtau_top(model_scores, model_scores_true),
                "clup": utils.clusters_p(model_scores),
                "evalcount_smooth": utils.evalcount_smooth(
                    model_scores, model_scores_true, budget
                ),
                "evalcount_top": utils.evalcount_top(
                    model_scores, model_scores_true, budget
                ),
            }
        )
    return [result | {"data_name": data_name} for result in output]


def simulate_subset2evaluate(
    data_names=None,
    fn_kwargs={},
    max_workers=None,
    seeds=1,
):
    print("Running", fn_kwargs)

    data_all = [
        (
            data_name,
            data,
            fn_kwargs,
            BUDGETS,
        )
        for data_name, data in utils.load_data(human_scores_only=False).items()
        if data_names is None or data_name in data_names
        for _ in range(seeds)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        output = [
            item
            for list in executor.map(_simulate_subset2evaluate, data_all)
            for item in list
        ]

    # aggregate across seeds
    data_agg = collections.defaultdict(list)
    for item in output:
        data_agg[(item["data_name"], item["budget"])].append(item)

    def compute_stats(xs):
        keys = [
            "tau",
            "wtau_smooth",
            "wtau_top",
            "clup",
            "evalcount_smooth",
            "evalcount_top",
        ]
        out = {
            "data_name": xs[0]["data_name"],
            "budget": xs[0]["budget"],
        }
        for key in keys:
            out[key] = np.mean([x[key] for x in xs])
            out[key + "_ci"] = utils.confidence_interval([x[key] for x in xs])
        return out

    return [compute_stats(cs) for cs in data_agg.values()]


def _simulate_subset2evaluate_perbudget(args):
    (
        data_name,
        data,
        fn_kwargs,
        BUDGETS,
    ) = args
    import subset2evaluate.select_subset

    budgets = [int(len(data) * len(data[0]["scores"]) * b) for b in BUDGETS]
    # hotfix for missing values
    for item in data:
        for model in item["scores"]:
            item["scores"][model]["MetricX-25"] = item["scores"][model].get(
                "MetricX-25", 0
            )
    model_scores_true = {
        model: [item["scores"][model]["human"] for item in data]
        for model in data[0]["scores"]
    }
    output = []
    for budget_p, budget in zip(BUDGETS, budgets):
        data_s2e = subset2evaluate.select_subset.basic(
            data, budget=math.ceil(budget_p * len(data)), **fn_kwargs
        )
        cost = 0
        model_scores = collections.defaultdict(list)
        for item in data_s2e:
            for model, model_score in item["scores"].items():  # type: ignore
                model_scores[model].append(model_score["human"])
                cost += 1
                if cost > budget:
                    break
            if cost > budget:
                break
        output.append(
            {
                "budget": budget_p,
                "tau": utils.tau(model_scores, model_scores_true),
                "wtau_smooth": utils.wtau_smooth(model_scores, model_scores_true),
                "wtau_top": utils.wtau_top(model_scores, model_scores_true),
                "clup": utils.clusters_p(model_scores),
                "evalcount_smooth": utils.evalcount_smooth(
                    model_scores, model_scores_true, budget
                ),
                "evalcount_top": utils.evalcount_top(
                    model_scores, model_scores_true, budget
                ),
            }
        )
    return [result | {"data_name": data_name} for result in output]


def simulate_subset2evaluate_perbudget(
    data_names=None,
    fn_kwargs={},
    max_workers=None,
):
    print("Running", fn_kwargs)
    data_all = [
        (
            data_name,
            data,
            fn_kwargs,
            BUDGETS,
        )
        for data_name, data in utils.load_data(human_scores_only=False).items()
        if data_names is None or data_name in data_names
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        output = [
            item
            for list in executor.map(_simulate_subset2evaluate_perbudget, data_all)
            for item in list
        ]

    # aggregate across seeds
    data_agg = collections.defaultdict(list)
    for item in output:
        data_agg[(item["data_name"], item["budget"])].append(item)

    def compute_stats(xs):
        keys = [
            "tau",
            "wtau_smooth",
            "wtau_top",
            "clup",
            "evalcount_smooth",
            "evalcount_top",
        ]
        out = {
            "data_name": xs[0]["data_name"],
            "budget": xs[0]["budget"],
        }
        for key in keys:
            out[key] = np.mean([x[key] for x in xs])
            out[key + "_ci"] = utils.confidence_interval([x[key] for x in xs])
        return out

    return [compute_stats(cs) for cs in data_agg.values()]
