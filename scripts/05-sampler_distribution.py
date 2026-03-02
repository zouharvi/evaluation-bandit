# %%

from evaluation_bandit import algorithms, utils, estimators, utils_fig
import random
import matplotlib.pyplot as plt
import numpy as np
import collections
import statistics


def shuffled(ys):
    ys = list(ys)
    random.shuffle(ys)
    return ys


data_all = utils.data_humanscores_only(utils.load_data())
KEYS = [
    ("wmt25", "cs-de_DE"),
    ("wmt25", "cs-uk_UA"),
]
data_all = {k: data_all[k] for k in KEYS}
for data in data_all.values():
    for line in data:
        line["cost"] = 1

# %%
import importlib

importlib.reload(algorithms)
BUDGETS = np.linspace(0.1, 1.0, 20, dtype=float)
results = collections.defaultdict(list)


for data in data_all.values():
    budgets = [int(p * len(data) * len(data[0]["scores"])) for p in BUDGETS]
    model_scores_true = utils.items_to_model_scores(data)
    model_estimates_true = estimators.mean(model_scores_true)
    model_ranking_true = utils.model_estimates_to_model_ranking(model_estimates_true)
    model_ranking_true_rev = {v: k for k, v in model_ranking_true.items()}

    results_local = collections.defaultdict(list)

    def distribution_aggregate(
        model_scores_all: list[utils.ModelScores],
    ) -> list[int]:
        dist_all = [
            [
                len(model_scores[model_ranking_true_rev[rank]])
                for rank in range(len(model_scores))
            ]
            for model_scores in model_scores_all
        ]
        dist = np.mean(dist_all, axis=0)
        # map to fixed 15 buckets
        i_to_bucket = [int(i / len(dist) * 15) for i in range(len(dist))]
        bucket_dist = collections.defaultdict(list)
        for i, d in zip(range(len(dist)), dist):
            bucket_dist[i_to_bucket[i]].append(d)
        return [statistics.mean(bucket) for bucket in bucket_dist.values()]

    for _ in range(5):
        data = shuffled(data)
        results_local["uniform"] += algorithms.uniform(data, budgets=budgets)
        results_local["weighted_sampling"] += algorithms.weighted_sampling(
            data, budgets=budgets
        )
        results_local["greedy_oracle"] += algorithms.greedy_oracle_invariant(
            data, budgets=budgets, batch_size=10
        )
    results["uniform"].append(distribution_aggregate(results_local["uniform"]))
    results["weighted_sampling"].append(
        distribution_aggregate(results_local["weighted_sampling"])
    )
    results["greedy_oracle"].append(
        distribution_aggregate(results_local["greedy_oracle"])
    )

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
for ax, (algorithm, dists_all) in zip(axs, results.items()):
    dist = np.mean(dists_all, axis=0)
    ax.bar(range(len(dist)), dist)
    # ax.set_ylim(0, 100)
    ax.set_title(algorithm)

plt.tight_layout()
plt.show()
