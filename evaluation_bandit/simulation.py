import collections
from typing import Callable
from evaluation_bandit import utils, estimators
import numpy as np
import concurrent.futures
import tqdm


def _simulate(args):
    (
        seed,
        data_name,
        data,
        fn,
        kwargs_fn,
        estimator_fn,
        BUDGETS,
        objectives_extra,
    ) = args
    _sum_cost = sum(item["cost"] for item in data)
    _models = len(data[0]["scores"])
    budgets = [int(_sum_cost * _models * b) for b in BUDGETS]
    if "budgets" in fn.__code__.co_varnames:
        model_scores_all = fn(data, budgets=budgets, **kwargs_fn)
    elif "budget" in fn.__code__.co_varnames:
        model_scores_all = [fn(data, budget=budget, **kwargs_fn) for budget in budgets]
    else:
        raise ValueError(f"Function {fn.__name__} does not accept budget nor budgets")
    model_scores_true = {
        model: [item["scores"][model] for item in data] for model in data[0]["scores"]
    }
    model_estimates_true = estimator_fn(model_scores_true)
    output = []
    for budget_p, budget, model_scores in zip(BUDGETS, budgets, model_scores_all):
        model_estimates = estimator_fn(model_scores)
        model_estimates_count = {
            x[0]: x[1]
            for x in sorted(
                estimators.count(model_scores).items(),
                key=lambda x: model_estimates[x[0]],
                reverse=True,
            )
        }
        output.append(
            {
                "budget": budget_p,
                "wtau": utils.wtau(model_estimates, model_estimates_true),
                "model_estimates": model_estimates,
                "model_estimates_count": model_estimates_count,
            }
        )
        if "tau" in objectives_extra:
            output[-1]["tau"] = utils.tau(model_estimates, model_estimates_true)
        if "evalfocus" in objectives_extra:
            output[-1]["evalfocus"] = utils.evalfocus(model_scores, model_scores_true)
        if "avg_pval" in objectives_extra:
            output[-1]["avg_pval"] = utils.avg_pval(model_scores)
        if "avg_payoff" in objectives_extra:
            output[-1]["avg_payoff"] = utils.avg_payoff(model_scores)
        if "clusters" in objectives_extra:
            output[-1]["clusters"] = utils.model_clusters(model_scores)
        if "wtau_pow1" in objectives_extra:
            output[-1]["wtau_pow1"] = utils.wtau_pow(
                model_estimates, model_estimates_true, 1
            )
        if "wtau_pow2" in objectives_extra:
            output[-1]["wtau_pow2"] = utils.wtau_pow(
                model_estimates, model_estimates_true, 2
            )
        if "wtau_pow05" in objectives_extra:
            output[-1]["wtau_pow05"] = utils.wtau_pow(
                model_estimates, model_estimates_true, 0.5
            )
        if "wtau_top3" in objectives_extra:
            output[-1]["wtau_top3"] = utils.wtau_topk(
                model_estimates, model_estimates_true, 3
            )
        if "wtau_top1" in objectives_extra:
            output[-1]["wtau_top1"] = utils.wtau_topk(
                model_estimates, model_estimates_true, 1
            )
        if "wtau_bot3" in objectives_extra:
            output[-1]["wtau_bot3"] = utils.wtau_botk(
                model_estimates, model_estimates_true, 3
            )
        if "wtau_middle3" in objectives_extra:
            output[-1]["wtau_middle3"] = utils.wtau_middlek(
                model_estimates, model_estimates_true, 3
            )
        if "wtau_revpow1" in objectives_extra:
            output[-1]["wtau_revpow1"] = utils.wtau_revpow(
                model_estimates, model_estimates_true, 1
            )
    return [result | {"data_name": data_name, "seed": seed} for result in output]


def simulate(
    fn: Callable,
    seeds=1,
    kwargs_fn={},
    data_all_fn=utils.load_data,
    data_sorter_fn=None,
    estimator_fn=estimators.mean,
    max_workers=None,
    cache_data_sorter=True,
    objectives_extra=None,
):
    print("Running", fn.__name__, "with", kwargs_fn)

    BUDGETS = np.linspace(0.1, 1.0, 20, dtype=float)

    data_all = data_all_fn()
    data_all_len = len(data_all)
    if cache_data_sorter:
        # compute sorter only once
        data_all = (
            (
                data_name,
                utils.data_humanscores_only(data_sorter_fn(data)),
                fn,
                kwargs_fn,
                estimator_fn,
                BUDGETS,
            )
            for data_name, data in data_all.items()
        )
        data_all = ((seed, *tpl) for tpl in data_all for seed in range(seeds))
    else:
        # sometimes we want the stochasticity
        data_all = (
            (
                seed,
                data_name,
                utils.data_humanscores_only(data_sorter_fn(data)),
                fn,
                kwargs_fn,
                estimator_fn,
                BUDGETS,
                objectives_extra,
            )
            for data_name, data in data_all.items()
            for seed in range(seeds)
        )
    print("Running simulations")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        output = [
            item
            for list in tqdm.tqdm(
                executor.map(_simulate, data_all), total=data_all_len * seeds
            )
            for item in list
        ]

    # aggregate across seeds
    data_agg = collections.defaultdict(list)
    for item in output:
        data_agg[(item["data_name"], item["budget"])].append(item)

    def compute_stats(xs):
        keys_to_aggregate = (
            "tau",
            "wtau",
            "evalfocus",
            "avg_pval",
        )
        out = {
            "data_name": xs[0]["data_name"],
            "budget": xs[0]["budget"],
            "model_estimates_count": [x["model_estimates_count"] for x in xs],
        }
        if "stability" in objectives_extra:
            out["stability"] = utils.stability([x["model_estimates"] for x in xs])
        for key in keys_to_aggregate:
            if key not in xs[0]:
                continue
            out[key] = np.mean([x[key] for x in xs])
            out[key + "_ci"] = utils.confidence_interval([x[key] for x in xs])
        return out

    return [compute_stats(cs) for cs in data_agg.values()]


def subset2evaluate_to_sorter(cost_normalize=False, **kwargs_fn):
    def sorter(data):
        import subset2evaluate.select_subset

        # make sure unified-ish metric is present everywhere
        # combining different metrics is not an issue since each dataset has the same one
        for line in data:
            if not cost_normalize:
                line["cost"] = 1
            for model_v in line["scores"].values():
                x = 0
                for metric in ["MetricX-25", "MetricX-24", "MetricX-23", "chrF"]:
                    if metric in model_v:
                        x = model_v[metric]
                        break
                model_v["metric"] = x

        data = subset2evaluate.select_subset.basic(list(data), **kwargs_fn)
        # when we don't cost_normalize, this does nothing
        # make sure that utility is positive
        min_utility = min(x["subset2evaluate_utility"] for x in data)
        min_cost = min(x["cost"] for x in data)
        data.sort(
            key=lambda x: (x["subset2evaluate_utility"] - min_utility)
            / (x["cost"] - min_cost + 0.1),
            reverse=True,
        )

        data_by_domain = collections.defaultdict(list)
        for item in data:
            data_by_domain[item["domain"]].append(item)

        # interleave domains
        data_new = []
        while data_by_domain:
            for domain in list(data_by_domain.keys()):
                data_new.append(data_by_domain[domain].pop(0))
                if not data_by_domain[domain]:
                    data_by_domain.pop(domain)
        return data_new

    return sorter
