# %%

from evaluation_bandit import utils, utils_fig
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import beta, norm
import random

# %%

fig, axs = plt.subplots(5, 6, figsize=(10, 5))
data_all = utils.data_humanscores_only(utils.load_data())
data_all = {
    k: v
    for k, v in data_all.items()
    if len(v) > 15
    and statistics.variance(utils.items_to_model_scores(v, average=True).values()) >= 20
}

for (data_name, data), ax in zip(data_all.items(), axs.flat):
    scores_avg = utils.items_to_model_scores(data, average=True)
    data_std = list(scores_avg.values())
    counts, bins, _ = ax.hist(data_std, bins=range(0, 110, 10), color="black")
    a, b_param = norm.fit(data_std)
    x = np.linspace(0, 100, 100)
    y = norm.pdf(x, a, b_param) * len(data_std) * (bins[1] - bins[0])
    ax.plot(x, y, color="tab:red", linewidth=2)
    # data name
    ax.text(
        5,
        12,
        "/".join(data_name)
        .replace("-", r"${\rightarrow}$")
        .split("_")[0]
        .replace("wmt", "WMT"),
        fontsize=10,
    )

    ax.set_yticks([])

    if ax not in axs[-1, :]:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, 100])
        ax.set_xlabel(r"$\mu$", fontsize=10, labelpad=-7, rotation=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 16)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.5)
plt.savefig("../figures/histogram_models.svg")
plt.show()

# %%
data = [list(item["scores"].values()) for v in data_all.values() for item in v]
data = [
    item
    for item in data
    if statistics.variance(item) >= 20
    and statistics.variance(item) <= 200
    and len(item) >= 15
]
fig, axs = plt.subplots(10, 12, figsize=(10, 5))
for item_i, (item, ax) in enumerate(
    zip(random.Random(0).choices(data, k=12 * 10), axs.flat)
):
    ax.hist(item, color="black", bins=range(0, 110, 10))
    a, b_param, loc, scale = beta.fit(item, floc=-0.001, fscale=100.002)
    x = np.linspace(0, 100, 100)
    y = beta.pdf(x, a, b_param, loc, scale) * len(item) * 10
    ax.plot(x, y, color="tab:purple", linewidth=2)
    ax.text(5, 7, f"Item {item_i + 1}", fontsize=10)

    ax.set_yticks([])

    if ax not in axs[-1, :]:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, 100], ["   0", "100      "])
        ax.set_xlabel(r"$\mu$  ", fontsize=10, labelpad=-7, rotation=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.5)
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.savefig("../figures/histogram_items.svg")
plt.show()

# %%


fig, axs = plt.subplots(5, 6, figsize=(10, 5))

for (data_name, data), ax in zip(data_all.items(), axs.flat):
    model_scores = utils.items_to_model_scores(data, average=False)
    model_scores = {
        model: (statistics.mean(scores), statistics.stdev(scores))
        for model, scores in model_scores.items()
    }
    data_std = list([v[1] for v in model_scores.values()])
    data_mean = list([v[0] for v in model_scores.values()])
    ax.scatter(data_mean, data_std, color="black")
    # counts, bins, _ = ax.hist(data_std, bins=range(0, 40, 5), color="black")
    # data name
    ax.text(
        5,
        40,
        "/".join(data_name)
        .replace("-", r"${\rightarrow}$")
        .split("_")[0]
        .replace("wmt", "WMT"),
        fontsize=10,
    )

    if ax not in axs[:, 0]:
        ax.set_yticks([])
    else:
        ax.set_yticks([0, 50])
        ax.set_ylabel(r"$\sigma$", fontsize=10, labelpad=-7, rotation=0)
    if ax not in axs[-1, :]:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, 100])
        ax.set_xlabel(r"$\mu$", fontsize=10, labelpad=-7, rotation=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.5)
plt.savefig("../figures/histogram_models_variance.svg")
plt.show()
