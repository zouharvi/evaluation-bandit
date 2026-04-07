# %%
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("../figures/microplot_weight", exist_ok=True)


def plot_microplot(name, fn):
    # Data generation
    x_dots = np.arange(1, 15)
    y_dots = fn(x_dots)

    # Plotting
    fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

    # Little dots on natural numbers
    ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

    # No decorations, Tufte style (remove spines and ticks)
    ax.axis("off")

    # Save as SVG
    plt.tight_layout(pad=0)
    plt.ylim(-0.1, 1.1)
    plt.savefig(f"../figures/microplot_weight/{name}.svg", transparent=True)
    plt.show()


plot_microplot("pow1", lambda x: 1 / x)
plot_microplot("pow2", lambda x: 1 / x**2)
plot_microplot("pow3", lambda x: 1 / x**3)
plot_microplot("top3", lambda x: [1 if i <= 3 else 0 for i in x])
plot_microplot("top1", lambda x: [1 if i == 1 else 0 for i in x])
plot_microplot("bot3", lambda x: [1 if i >= len(x) - 2 else 0 for i in x])
plot_microplot("middle3", lambda x: [1 if 6 <= i <= 8 else 0 for i in x])
plot_microplot("pow0.5", lambda x: 1 / x**0.5)
plot_microplot("revpow1", lambda x: 1 / (len(x) - x + 1))
plot_microplot(
    "random", lambda x: sorted(np.random.default_rng(0).random(len(x)), reverse=True)
)


# %%

import evaluation_bandit.utils_fig
import evaluation_bandit.utils
import statistics

data_all = evaluation_bandit.utils.load_data()
data_all = [
    [
        statistics.mean([item["scores"][model] for item in data])
        for model in data[0]["scores"].keys()
    ]
    for data in data_all.values()
]

data_all = [
    [100 + 4 * x for x in data] if statistics.mean(data) < 0 else data
    for data in data_all
]
data_all = [x for data in data_all for x in data]

fig, ax = plt.subplots(figsize=(1, 0.3))
plt.hist(data_all, color="black")
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../figures/microplot_model_avg.svg", transparent=True)
plt.show()
