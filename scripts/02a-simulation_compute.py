 # pyright: ignore[reportRedeclaration]

from evaluation_bandit import simulation, algorithms, estimators
import argparse
import math
import statistics
import os
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--method",
    type=str,
    required=True,
)
args.add_argument(
    "--method-sorter",
    type=str,
    default="random",
)
args.add_argument(
    "--method-estimator",
    type=str,
    default="mean",
    choices=["mean", "additive", "count"],
)
args.add_argument(
    "--method-estimator-eval",
    type=str,
    default="mean",
    choices=["mean", "additive", "count"],
)
args.add_argument(
    "--seeds",
    type=int,
    default=100,
)
args.add_argument(
    "--max-workers",
    type=int,
    default=15,
)
args = args.parse_args()

if args.method_sorter == "random":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(method="random")
elif args.method_sorter == "random_cost":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="random", cost_normalize=True
    )
elif args.method_sorter == "metricvar":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_var", metric="metric"
    )
elif args.method_sorter == "metricavg":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="metric"
    )
elif args.method_sorter == "humanavg":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="human"
    )
elif args.method_sorter == "metricavg_cost":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="metric", cost_normalize=True
    )
elif args.method_sorter == "humanavg_cost":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_avg", metric="human", cost_normalize=True
    )
elif args.method_sorter == "metriccons":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="metric_cons", metric="metric"
    )
elif args.method_sorter == "diversity_bleu":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="BLEU"
    )
elif args.method_sorter == "diversity_unigram":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="unigram"
    )
elif args.method_sorter == "diversity_lm":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="diversity", metric="lm"
    )
elif args.method_sorter == "cometconfidence":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="comet_instant_confidence"
    )
elif args.method_sorter == "sentinel_mqm":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(method="sentinel_src_mqm")
elif args.method_sorter == "precomet_diffdisc":
    data_sorter_fn = simulation.subset2evaluate_to_sorter(
        method="precomet_diffdisc_direct"
    )
else:
    raise ValueError(f"Unknown method_sorter: {args.method_sorter}")


if args.method_estimator == "mean":
    estimator_fn = estimators.mean
elif args.method_estimator == "additive":
    estimator_fn = estimators.additive
elif args.method_estimator == "count":
    estimator_fn = estimators.count
else:
    raise ValueError(f"Unknown method_estimator: {args.method_estimator}")


if args.method_estimator_eval == "mean":
    estimator_eval_fn = estimators.mean
elif args.method_estimator_eval == "additive":
    estimator_eval_fn = estimators.additive
elif args.method_estimator_eval == "count":
    estimator_eval_fn = estimators.count
else:
    raise ValueError(f"Unknown method_estimator_eval: {args.method_estimator_eval}")


def simulate(fn, kwargs_fn={}, **kwargs):
    return simulation.simulate(
        fn=fn,
        kwargs_fn=kwargs_fn,
        seeds=args.seeds,
        data_sorter_fn=data_sorter_fn,
        max_workers=args.max_workers,
        cache_data_sorter=args.method_sorter != "random",
        estimator_fn=estimator_eval_fn,
        **kwargs,
    )


if args.method == "uniform":
    output = simulate(algorithms.uniform)
elif args.method == "uniform_nonsquare":
    output = simulate(algorithms.uniform_nonsquare)
elif args.method == "greedy_oracle":
    output = simulate(
        algorithms.greedy_oracle, kwargs_fn=dict(batch_size=25, batch_size_lookahead=75)
    )
elif args.method == "greedy_oracle_invariant":
    output = simulate(
        algorithms.greedy_oracle_invariant,
        kwargs_fn=dict(shuffle_repetitions=10, batch_size=25, batch_size_lookahead=75),
    )
elif args.method == "successive_rejects_constant":
    output = simulate(algorithms.successive_rejects, kwargs_fn=dict(phases="constant"))
elif args.method == "weighted_sampling_rank":

    def sampling_fn(ys, rank, total):
        return 1 / (rank + 1)

    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=sampling_fn,
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_rankpow2":

    def sampling_fn(ys, rank, total):
        return 1 / ((rank + 1) ** 2)

    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=sampling_fn,
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_bolzmann":

    def sampling_fn(ys, rank, total, temperature=10):
        return math.exp(statistics.mean(ys) / temperature)

    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=sampling_fn,
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_epsilongreedy":

    def sampling_fn(ys, rank, total, epsilon=0.5):
        return 1 / (rank + 1) if rank == 0 else epsilon / (total - 1)

    output = simulate(
        algorithms.weighted_sampling,
        kwargs_fn=dict(
            sampling_fn=sampling_fn,
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "weighted_sampling_oracle_rank":

    def sampling_fn(ys, rank, total):
        return 1 / (rank + 1)

    output = simulate(
        algorithms.weighted_sampling_oracle,
        kwargs_fn=dict(sampling_fn=sampling_fn),
    )
elif args.method == "weighted_sampling_oracle_rankpow2":

    def sampling_fn(ys, rank, total):
        return 1 / ((rank + 1) ** 2)

    output = simulate(
        algorithms.weighted_sampling_oracle,
        kwargs_fn=dict(sampling_fn=sampling_fn),
    )
elif args.method == "confusion_minimization":
    def sampling_fn(ys, rank, total):
        return 1 / (rank + 1)**2

    output = simulate(
        algorithms.confusion_minimization,
        kwargs_fn=dict(
            sampling_fn=sampling_fn,
            estimator_fn=estimator_fn,
        ),
    )
elif args.method == "ucb":
    output = simulate(
        algorithms.upper_confidence_bound,
        kwargs_fn=dict(topk=1, c=100 * math.sqrt(2), estimator_fn=estimator_fn),
    )
elif args.method == "ambiguity_reduction_11":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=1, weight_pairwise=1),
    )
elif args.method == "ambiguity_reduction_01":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=0, weight_pairwise=1),
    )
elif args.method == "ambiguity_reduction_10":
    output = simulate(
        algorithms.statistical_ambiguity_reduction,
        kwargs_fn=dict(weight_pointwise=1, weight_pairwise=0),
    )
elif args.method == "successive_halving":
    output = simulate(algorithms.successive_halving)
elif args.method == "pvalue_rejects":
    output = simulate(
        algorithms.pvalue_rejects,
    )
elif args.method == "thompson_sampling":
    output = simulate(
        algorithms.thompson_sampling,
        kwargs_fn=dict(estimator_fn=estimator_fn),
    )
else:
    raise ValueError(f"Unknown method: {args.method}")

for key in ["wtau", "evalfocus", "tau", "avg_pval"]:
    print(key, statistics.mean([x[key] for x in output]))

os.makedirs("computed/02/", exist_ok=True)
with open(
    f"computed/02/{args.method}#{args.method_sorter}#{args.method_estimator}#{args.method_estimator_eval}.json",
    "w",
) as f:
    json.dump(output, f)
