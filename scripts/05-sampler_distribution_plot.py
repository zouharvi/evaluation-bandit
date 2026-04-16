# %%

import json
import numpy as np
import math
import statistics
import collections
from evaluation_bandit import utils_fig
import matplotlib.pyplot as plt

METHODS = [
    {"method_latex": "Uniform", "method": "uniform_nonsquare", "color": "black"},
    {
        "method_latex": "Sampling rank",
        "method": "weighted_sampling_rank",
        "color": utils_fig.COLORS[1],
    },
    {
        "method_latex": "Greedy oracle",
        "method": "greedy_oracle_invariant_wtau_pow2",
        "color": utils_fig.COLORS[4],
    },
    {
        "method_latex": "Upper confidence bound",
        "method": "ucb",
        "color": utils_fig.COLORS[2],
    },
    {
        "method_latex": "Confusion minimization",
        "method": "confusion_minimization",
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


plt.figure(figsize=(3.5, 2))

for method in METHODS:
    with open(f"../computed/02/{method['method']}#random#mean#mean.json", "r") as f:
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
        linewidth=0.5,
        color=method["color"],
        marker="o",
        markersize=3,
    )

# legend above plot
plt.xticks([0, 2, 4, N_BINS], ["1st", "3rd", "...", "last"])
plt.ylabel(r"Number of items $|R_m|$")
plt.xlabel(r"rank${}_m$", labelpad=-5)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0)
box = plt.gca().get_position()
handles, labels = plt.gca().get_legend_handles_labels()

plt.gca().set_facecolor("none")
plt.gcf().patch.set_facecolor("none")

plt.savefig("../figures/sampler_distribution.svg")
plt.show()

# plot only legend
fig_legend = plt.figure(figsize=(2.5, 1.2))

# make the lines three times as thick
for handle in handles:
    handle.set_linewidth(6.0)  # type: ignore

plt.legend(
    handles,
    labels,
    frameon=False,
    loc="center",
    ncols=1,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=1,
)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../figures/sampler_distribution_legend.svg")
plt.show()
