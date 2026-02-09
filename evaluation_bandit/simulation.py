import collections
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
    seeds=1,
    fn_kwargs={},
    accepts_budgets=False,
    ranking_only=False,
    fn_data_all=utils.load_data,
    fn_data_sorter=None,
    max_workers=None,
):
    print("Running", fn.__name__, "with", fn_kwargs)

    data_all = [
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
        for data_name, data in fn_data_all().items()
        for seed in range(seeds)
    ]
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
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


def subset2evaluate_to_sorter(**fn_kwargs):
    def sorter(data):
        import subset2evaluate.select_subset

        # make sure MetricX-25 is present everywhere
        for line in data:
            for model_v in line["scores"].values():
                model_v["MetricX-25"] = model_v.get("MetricX-25", 0)

        data = subset2evaluate.select_subset.basic(data, **fn_kwargs)

        data_by_domain = collections.defaultdict(list)
        for item in data:
            data_by_domain[item["domain"]].append(item)
        data_new = []

        # interleave domains
        while data_by_domain:
            for domain in list(data_by_domain.keys()):
                data_new.append(data_by_domain[domain].pop(0))
                if not data_by_domain[domain]:
                    data_by_domain.pop(domain)
        return data_new

    return sorter
