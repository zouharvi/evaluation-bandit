import collections
from typing import Callable
from evaluation_bandit import utils
import numpy as np
import concurrent.futures
import tqdm
import itertools


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
                    "wtau": utils.wtau(model_scores, model_scores_true),
                    "evalfocus": utils.evalfocus(
                        model_scores, model_scores_true, budget
                    ),
                    "tau": utils.tau(model_scores, model_scores_true),
                    "avg_pval": utils.avg_pval(model_scores),
                    "model_scores": model_scores,
                }
            )
        else:
            raise NotImplementedError
    return [result | {"data_name": data_name, "seed": seed} for result in output]


def simulate(
    fn: Callable,
    seeds=1,
    fn_kwargs={},
    accepts_budgets=False,
    ranking_only=False,
    fn_data_all=utils.load_data,
    fn_data_sorter=None,
    max_workers=None,
    cache_data_sorter=True,
):
    print("Running", fn.__name__, "with", fn_kwargs)

    # BUDGETS = np.linspace(0.1, 0.9, 20, dtype=float)
    BUDGETS = np.linspace(0.1, 1.0, 20, dtype=float)

    data_all = fn_data_all()
    data_all_len = len(data_all)
    if cache_data_sorter:
        # compute sorter only once
        data_all = (
            (
                data_name,
                utils.data_humanscores_only(fn_data_sorter(data)),
                fn,
                fn_kwargs,
                accepts_budgets,
                ranking_only,
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
                utils.data_humanscores_only(fn_data_sorter(data)),
                fn,
                fn_kwargs,
                accepts_budgets,
                ranking_only,
                BUDGETS,
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
        keys = (
            "wtau",
            "evalfocus",
            "tau",
            "avg_pval",
        )
        out = {
            "data_name": xs[0]["data_name"],
            "budget": xs[0]["budget"],
        }
        out["stability"] = utils.stability([x["model_scores"] for x in xs])
        for key in keys:
            out[key] = np.mean([x[key] for x in xs])
            out[key + "_ci"] = utils.confidence_interval([x[key] for x in xs])
        return out

    return [compute_stats(cs) for cs in data_agg.values()]


def subset2evaluate_to_sorter(**fn_kwargs):
    def sorter(data):
        import subset2evaluate.select_subset

        # make sure MetricX-25 is present everywhere
        for line in data:
            for model_v in line["scores"].values():
                x = 0
                for metric in ["MetricX-25", "MetricX-24", "MetricX-23", "chrF"]:
                    if metric in model_v:
                        x = model_v[metric]
                        break
                model_v["metric"] = x

        data = subset2evaluate.select_subset.basic(list(data), **fn_kwargs)

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
