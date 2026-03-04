# %%

import json
import numpy as np
import math
import statistics
import collections
from evaluation_bandit import utils_fig
import matplotlib.pyplot as plt

METHODS = [
    {"method_latex": "Uniform", "method": "uniform", "color": "black"},
    {
        "method_latex": "Greedy Oracle",
        "method": "greedy_oracle_invariant",
        "color": utils_fig.COLORS[4],
    },
    {
        "method_latex": "Weighted Sampling (rank)",
        "method": "weighted_sampling_rank",
        "color": utils_fig.COLORS[1],
    },
    {
        "method_latex": "Upper Confidence Bound",
        "method": "ucb",
        "color": utils_fig.COLORS[3],
    },
]
N_BINS = 15


def digitize(ys: list[dict[str, int]]):
    dist = np.mean([list(y.values()) for y in ys], axis=0)
    bucket_dist = collections.defaultdict(list)
    for i, d in zip(range(len(dist)), dist):
        bucket_dist[math.floor(i / (len(dist) - 1) * N_BINS)].append(d)
    return [statistics.mean(bucket) for bucket in bucket_dist.values()]


plt.figure(figsize=(3, 2))

for method in METHODS:
    with open(f"../computed/04-x/{method['method']}#random#mean#mean.json", "r") as f:
        data = json.load(f)
    ys = [
        digitize(x["model_estimates_count"])
        for x in data
        if len(x["model_estimates_count"][0]) >= N_BINS + 1
    ]
    ys = np.mean(ys, axis=0)
    plt.plot(
        ys,
        label=method["method_latex"],
        linewidth=2,
        color=method["color"],
        marker="o",
        markersize=4,
    )

# legend above plot
plt.xticks([0, 2, 4, N_BINS], ["1st", "3rd", "...", "last"])
plt.ylabel(r"Number of items $|R_m|$")
plt.xlabel(r"rank${}_m$")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0)
box = plt.gca().get_position()
handles, labels = plt.gca().get_legend_handles_labels()
plt.show()

# plot only legend
fig_legend = plt.figure(figsize=(3, 0.5))

# make the lines three times as thick
for handle in handles:
    handle.set_linewidth(6.0)

plt.legend(handles, labels, frameon=False, loc="center", ncols=2)
plt.axis("off")
plt.tight_layout()
plt.show()
