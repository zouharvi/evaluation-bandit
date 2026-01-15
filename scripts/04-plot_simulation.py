# %%

from translation_bandit import simulation, algorithms
import math
import statistics

SEEDS = 100

outputs_baseline = simulation.simulate(
    fn=algorithms.baseline,
    seeds=SEEDS,
)
outputs_successive_rejects_constant = simulation.simulate(
    fn=algorithms.successive_rejects,
    fn_kwargs=dict(phases="constant"),
    seeds=SEEDS,
)


def sampling_fn_ranksmooth(ys, rank, total):
    return 1 / (rank + 1)


outputs_stochastic_sampling_ranksmooth = simulation.simulate(
    fn=algorithms.weighted_sampling,
    fn_kwargs=dict(sampling_fn=sampling_fn_ranksmooth),
    accepts_budgets=True,
    seeds=SEEDS,
)


def sampling_fn_bolzmann(ys, rank, total, temperature=1):
    return math.exp(statistics.mean(ys) / temperature)


outputs_stochastic_sampling_bolzmann = simulation.simulate(
    fn=algorithms.weighted_sampling,
    fn_kwargs=dict(sampling_fn=sampling_fn_bolzmann),
    accepts_budgets=True,
    seeds=SEEDS,
)

# %%
outputs_pointwise_pairwise_ambiguity = simulation.simulate(
    fn=algorithms.pointwise_pairwise_ambiguity,
    fn_kwargs=dict(weight_pointwise=1, weight_pairwise=1),
    accepts_budgets=True,
    seeds=SEEDS,
)

# %%
import functools

outputs_stochastic_sampling_bolzmann_multi = {}
for temperature in [0.3, 0.4, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5, 10]:
    outputs_stochastic_sampling_bolzmann_multi[temperature] = simulation.simulate(
        fn=algorithms.weighted_sampling,
        fn_kwargs=dict(
            sampling_fn=functools.partial(sampling_fn_bolzmann, temperature=temperature)
        ),
        accepts_budgets=True,
        seeds=SEEDS,
    )


# %%

from translation_bandit import utils_fig
import collections
import numpy as np
import matplotlib.pyplot as plt
import importlib

importlib.reload(utils_fig)


def plot_output(outputs, label, axs, color=None):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    xs = [xs[0]["budget"] for xs in data_by_budget]
    for ax, key in zip(
        axs,
        ["tau", "wtau_smooth", "clup", "evalcount_smooth"],
    ):
        ax.plot(
            xs,
            [np.mean([x[key] for x in xs]) for xs in data_by_budget],
            label=label,
            color=color,
            linewidth=2.0,
            zorder=2 if label == "Random" else 1,
        )
        ax.fill_between(
            xs,
            [np.mean([x[key + "_ci"][0] for x in xs]) for xs in data_by_budget],
            [np.mean([x[key + "_ci"][1] for x in xs]) for xs in data_by_budget],
            alpha=0.4,
            color=color,
            linewidth=0.0,
        )


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 4.5))
axs = axs.flatten()

plot_output(outputs_successive_rejects_constant, "Successive rejects", axs, color="C0")
plot_output(
    outputs_pointwise_pairwise_ambiguity,
    r"Point-$&$pairwise ambiguity",
    axs,
    color="C3",
)
plot_output(
    outputs_stochastic_sampling_ranksmooth, "Sampling rank-based", axs, color="C1"
)
plot_output(outputs_stochastic_sampling_bolzmann, "Sampling Bolzmann", axs, color="C2")
plot_output(outputs_baseline, "Random", axs, color="black")

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))  # type: ignore
    ax.set_xlabel("Budget proportion")


def format_ax_label(ax, x, y, text):
    # \uparrow in top left of axis
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
    )


format_ax_label(axs[0], 0.02, 0.90, r"Standard $\tau$ $\uparrow$")
format_ax_label(axs[1], 0.02, 0.90, r"Weighted $\tau$ $\uparrow$")
format_ax_label(axs[2], 0.45, 0.90, r"Average $p$-value $\downarrow$")
format_ax_label(axs[3], 0.45, 0.20, r"Evaluation focus $\uparrow$")

axs[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
axs[0].set_ylim(0.85, 1.0 + 0.01)
axs[1].set_ylim(0.85, 1.0 + 0.01)
axs[2].set_ylim(None, 0.6)

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation_wmt25.svg")
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(8, 0.4))
handles, labels = axs[0].get_legend_handles_labels()
fig_legend.legend(
    handles,
    labels,
    loc="center",
    ncol=3,
    frameon=False,
    handlelength=1,
    handletextpad=0.3,
    columnspacing=1,
)
fig_legend.tight_layout(pad=0)
plt.axis("off")
plt.savefig("../figures/simulation_wmt25_legend.svg")
plt.show()

# %%
import os
import pickle

os.makedirs("cache/", exist_ok=True)
with open("cache/outputs_pointwise_pairwise_ambiguity.pkl", "wb") as f:
    pickle.dump(outputs_pointwise_pairwise_ambiguity, f)

# %%

import pickle

with open("cache/outputs_pointwise_pairwise_ambiguity.pkl", "rb") as f:
    outputs_pointwise_pairwise_ambiguity = pickle.load(f)
