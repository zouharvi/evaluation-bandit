# %%

from evaluation_bandit import algorithms, utils, estimators, utils_fig
import random
import matplotlib.pyplot as plt
import numpy as np
import collections
import statistics
import math
import concurrent
import os
import pickle
import importlib

importlib.reload(algorithms)


data_all = utils.data_humanscores_only(utils.load_data())
KEYS = [
    ("wmt25", "cs-de_DE"),
    # ("wmt25", "cs-uk_UA"),
]
data_all = {k: data_all[k] for k in KEYS}
for data in data_all.values():
    for line in data:
        line["cost"] = 1

BUDGETS = np.linspace(0.1, 1.0, 20, dtype=float)
results = collections.defaultdict(list)


def run_simulation(args):
    data, seed, budgets, model_ranking_true_rev = args
    data_local = list(data)
    random.Random(seed).shuffle(data_local)

    results_local = collections.defaultdict(list)
    results_local["uniform"] += algorithms.uniform(data_local, budgets=budgets)
    results_local["weighted_sampling"] += algorithms.weighted_sampling(
        data_local, budgets=budgets
    )
    results_local["greedy_oracle_invariant"] += algorithms.greedy_oracle(
        data_local,
        budgets=budgets,
        batch_size=25,
        batch_size_lookahead=50,
    )
    results_local["ucb"] += algorithms.upper_confidence_bound(
        data_local,
        budgets=budgets,
        c=100 * math.sqrt(2),
    )

    def distribution_aggregate(model_scores_all: list[utils.ModelScores]) -> list[int]:
        dist_all = [
            [
                len(model_scores[model_ranking_true_rev[rank]])
                for rank in range(len(model_scores))
            ]
            for model_scores in model_scores_all
        ]
        dist = np.mean(dist_all, axis=0)
        bucket_dist = collections.defaultdict(list)
        for i, d in zip(range(len(dist)), dist):
            bucket_dist[math.ceil(i / len(dist) * 15)].append(d)
        return [statistics.mean(bucket) for bucket in bucket_dist.values()]

    return {
        "uniform": [distribution_aggregate(results_local["uniform"])],
        "weighted_sampling": [
            distribution_aggregate(results_local["weighted_sampling"])
        ],
        "greedy_oracle_invariant": [
            distribution_aggregate(results_local["greedy_oracle_invariant"])
        ],
        "ucb": [distribution_aggregate(results_local["ucb"])],
    }


tasks = []
for data in data_all.values():
    budgets = [int(p * len(data) * len(data[0]["scores"])) for p in BUDGETS]
    model_scores_true = utils.items_to_model_scores(data)
    model_estimates_true = estimators.mean(model_scores_true)
    model_ranking_true = utils.model_estimates_to_model_ranking(model_estimates_true)
    model_ranking_true_rev = {v: k for k, v in model_ranking_true.items()}

    for seed in range(1):
        tasks.append((data, seed, budgets, model_ranking_true_rev))

with concurrent.futures.ProcessPoolExecutor() as executor:
    for res in executor.map(run_simulation, tasks):
        for k, v in res.items():
            results[k] += v

os.makedirs("computed/05", exist_ok=True)
with open("computed/05/computed.pkl", "wb") as f:
    pickle.dump(results, f)

# %%

fig, ax = plt.subplots(1, 1, figsize=(3, 2), sharey=True)
for algorithm, dists_all in results.items():
    dist = np.mean(dists_all, axis=0)
    ax.plot(
        range(len(dist)),
        dist,
        label=algorithm,
        linewidth=2,
    )
    # ax.set_ylim(0, 100)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r"rank${}_m$")
ax.set_ylabel(r"Number of items $|R_m|$")
ax.spines[["top", "right"]].set_visible(False)
# plt.legend()
plt.tight_layout()
plt.show()
