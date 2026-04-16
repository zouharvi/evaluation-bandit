# %%

import statistics
from evaluation_bandit import algorithms
from evaluation_bandit import simulation
from evaluation_bandit import estimators
from evaluation_bandit import utils
import os
import math
import functools
import argparse
import json

args = argparse.ArgumentParser()
args.add_argument(
    "--data", required=True, choices=["homo", "hetero", "binary", "likert"]
)
args.add_argument("--method", required=True)
args.add_argument("--estimator", default="mean", choices=["mean", "additive"])
args.add_argument("--seeds", default=100, type=int)
args.add_argument(
    "--max-workers",
    type=int,
    default=15,
)
args = args.parse_args()

if args.data == "homo":
    data_fn_loader = utils.load_data_synth_homo
elif args.data == "hetero":
    data_fn_loader = utils.load_data_synth_hetero
elif args.data == "binary":
    data_fn_loader = utils.load_data_synth_binary
elif args.data == "likert":
    data_fn_loader = utils.load_data_synth_likert
else:
    raise ValueError("Invalid data type")

kwargs_fn = {}

if args.estimator == "mean":
    estimator_fn = estimators.mean
elif args.estimator == "additive":
    estimator_fn = estimators.additive
else:
    raise ValueError("Invalid estimator")


if args.method == "uniform":
    fn = algorithms.uniform
elif args.method == "uniform_nonsquare":
    fn = algorithms.uniform_nonsquare
elif args.method.startswith("weighted_sampling_"):
    sampling_str = args.method.removeprefix("weighted_sampling_")
    if sampling_str == "rank":

        def sampling_fn(ys, rank, total):
            return 1 / (rank + 1)
    elif sampling_str == "rankpow2":

        def sampling_fn(ys, rank, total):
            return 1 / ((rank + 1) ** 2)
    elif sampling_str == "bolzmann":

        def sampling_fn(ys, rank, total, temperature=10):
            return math.exp(statistics.mean(ys) / temperature)

    elif sampling_str == "epsilongreedy":

        def sampling_fn(ys, rank, total, epsilon=0.5):
            return 1 / (rank + 1) if rank == 0 else epsilon / (total - 1)

    else:
        raise ValueError("Invalid sampling method")
    kwargs_fn["sampling_fn"] = sampling_fn
    kwargs_fn["estimator_fn"] = estimator_fn
    fn = algorithms.weighted_sampling
elif args.method == "ucb":
    fn = algorithms.upper_confidence_bound
    kwargs_fn["estimator_fn"] = estimator_fn
elif args.method == "confusion_minimization":

    def sampling_fn(ys, rank, total):
        return 1 / (rank + 1) ** 2

    kwargs_fn["sampling_fn"] = sampling_fn
    kwargs_fn["estimator_fn"] = estimator_fn
    fn = algorithms.confusion_minimization
elif args.method == "greedy_oracle_invariant_wtau_pow2":
    kwargs_fn["objective_fn"] = functools.partial(utils.wtau_pow, k=2)
    kwargs_fn["estimator_fn"] = estimator_fn
    fn = algorithms.greedy_oracle_invariant
elif args.method == "successive_rejects":
    fn = algorithms.successive_rejects
else:
    raise ValueError("Invalid method")


def data_fn():
    return {f"synth_{i}": data_fn_loader(seed=i) for i in range(args.seeds)}


outputs = simulation.simulate(
    fn=fn,
    kwargs_fn=kwargs_fn,
    seeds=1,
    data_all_fn=data_fn,
    data_sorter_fn=simulation.subset2evaluate_to_sorter(method="random"),
    max_workers=args.max_workers,
    cache_data_sorter=False,
    estimator_fn=estimator_fn,
    # objectives_extra=args.objectives_extra,
)

os.makedirs("computed/03/", exist_ok=True)
with open(
    f"computed/03/{args.data}#{args.method}#{args.estimator}.json",
    "w",
) as f:
    json.dump(outputs, f)
