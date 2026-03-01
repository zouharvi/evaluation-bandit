# %%

from evaluation_bandit import utils, utils_fig
import matplotlib.pyplot as plt
import numpy as np
import statistics
from scipy.stats import beta, norm
import random

# %%

fig, axs = plt.subplots(5, 6, figsize=(10, 5))
data_all = utils.load_data()
data_all = {
    k: v
    for k, v in data_all.items()
    if len(v) > 15
    and statistics.variance(
        utils.items_to_model_scores(
            utils.data_humanscores_only(v), average=True
        ).values()
    )
    >= 20
}

for (data_name, data), ax in zip(data_all.items(), axs.flat):
    data = utils.data_humanscores_only(data)
    scores_avg = utils.items_to_model_scores(data, average=True)
    data_avg = list(scores_avg.values())
    counts, bins, _ = ax.hist(data_avg, bins=range(0, 110, 10), color="black")
    a, b_param = norm.fit(data_avg)
    x = np.linspace(0, 100, 100)
    y = norm.pdf(x, a, b_param) * len(data_avg) * (bins[1] - bins[0])
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

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 16)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.5)
plt.savefig("../figures/histogram_averages.svg")
plt.show()

# %%
data = [
    list(item["scores"].values())
    for v in data_all.values()
    for item in utils.data_humanscores_only(v)
]
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

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.5)
plt.savefig("../figures/histogram_items.svg")
plt.show()
