# %%

import json


def read_computed(method, method_ranker):
    with open(f"../computed/04/{method}_{method_ranker}.json", "r") as f:
        return json.load(f)


outputs = [
    {
        "method_typst": "Uniform",
        "method_latex": "Uniform",
        "method": "uniform_nonsquare",
    },
    {
        "method_typst": "Successive rejects",
        "method_latex": "Successive rejects",
        "method": "successive_rejects_constant",
    },
    {
        "method_typst": "Sampling rank",
        "method_latex": "Sampling rank",
        "method": "weighted_sampling_rank",
    },
    {
        "method_typst": "Sampling rank-sqrt",
        "method_latex": "Sampling rank-sqrt",
        "method": "weighted_sampling_ranksqrt",
    },
    {
        "method_typst": "Sampling $epsilon$-Greedy",
        "method_latex": "Sampling $\\epsilon$-Greedy",
        "method": "weighted_sampling_epsilongreedy",
    },
    {
        "method_typst": "Sampling Bolzmann",
        "method_latex": "Sampling Bolzmann",
        "method": "weighted_sampling_bolzmann",
    },
    {
        "method_typst": "Upper Confidence Bound",
        "method_latex": "Upper Confidence Bound",
        "method": "ucb",
    },
    {
        "method_typst": "Ambiguity reduction $lambda$=$1$",
        "method_latex": "Ambiguity reduction $\\lambda=1$",
        "method": "ambiguity_reduction_11",
    },
    # no LaTeX for s2e
    {
        "method_typst": "Ambiguity reduction $lambda$=$0$",
        "method_latex": None,
        "method": "ambiguity_reduction_01",
    },
    {
        "method_typst": "Ambiguity reduction $lambda$=$infinity$",
        "method_latex": None,
        "method": "ambiguity_reduction_10",
    },
    # {"typst": "MetricVar", "latex": None, "method": "s2e_metricvar"},
    # {"typst": "MetricAvg", "latex": None, "method": "s2e_metricavg"},
    # {"typst": "MetricCons", "latex": None, "method": "s2e_metriccons"},
    # {"typst": "Diversity BLEU", "latex": None, "method": "s2e_diversity_bleu"},
    # {"typst": "Diversity Unigram", "latex": None, "method": "s2e_diversity_unigram"},
    # {"typst": "Diversity LM", "latex": None, "method": "s2e_diversity_lm"},
    # {"typst": "Instant confidence", "latex": None, "method": "s2e_cometconfidence"},
    # {"typst": "Sentinel MQM", "latex": None, "method": "s2e_sentinel_mqm"},
    # {"typst": "Pre-Comet DiffDisc", "latex": None, "method": "s2e_precomet_diffdisc"},
]

outputs = [
    output | {"method_ranker": method_ranker}
    for output in outputs
    for method_ranker in [
        "random",
        "metricavg",
        "metricvar",
        "metriccons",
        "diversity_bleu",
        "diversity_unigram",
        "diversity_lm",
        "cometconfidence",
        "sentinel_mqm",
        "precomet_diffdisc",
    ]
]

for output in outputs:
    try:
        output["data"] = read_computed(output["method"], output["method_ranker"])
    except FileNotFoundError:
        print(
            f"Warning: computed file for method {output['method']}_{output['method_ranker']} not found."
        )


from evaluation_bandit import utils_fig
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
        ["wtau", "evalfocus"],
    ):
        ax.plot(
            xs,
            [np.mean([x[key] for x in xs]) for xs in data_by_budget],
            label=label,
            color=color,
            linewidth=2.0,
            zorder=2 if label == "Uniform" else 1,
        )
        ax.fill_between(
            xs,
            [np.mean([x[key + "_ci"][0] for x in xs]) for xs in data_by_budget],
            [np.mean([x[key + "_ci"][1] for x in xs]) for xs in data_by_budget],
            alpha=0.4,
            color=color,
            linewidth=0.0,
        )


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 2.5))
# axs = axs.flatten()

output_i = 0
for output in outputs:
    if (
        output["method_latex"] is None
        or output["method_ranker"] != "random"
        or "data" not in output
    ):
        continue
    plot_output(
        output["data"],
        output["method_latex"],
        axs,
        color="black"
        if output["method"] == "uniform_nonsquare"
        else f"C{output_i - 1}",
    )
    output_i += 1

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))  # type: ignore
    ax.set_xlabel("Budget proportion")
    ax.set_xlim(0.1, 1.0)


axs[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
# axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
axs[0].set_ylim(0.83, 1.0 + 0.01)
axs[1].set_ylim(6.5, None)
axs[0].set_ylabel(r"Weighted $\tau$", labelpad=-5)
axs[1].set_ylabel("\nEvaluation focus", labelpad=1)

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation.svg")
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(8, 0.4))
handles, labels = axs[0].get_legend_handles_labels()
fig_legend.legend(
    handles,
    labels,
    loc="center",
    ncol=4,
    frameon=False,
    handlelength=1,
    handletextpad=0.3,
    columnspacing=1,
)
fig_legend.tight_layout(pad=0)
plt.axis("off")
plt.savefig("../figures/simulation_legend.svg")
plt.show()


# area under curve table


def area_under_curve(outputs, key):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    x = np.trapezoid(
        y=[
            np.mean([x[key] for x in xs])
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 0.9
        ],
        x=[
            xs[0]["budget"]
            for xs in data_by_budget
            if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 0.9
        ],
    ) / (1.0 - 0.1)
    if key == "clup":
        x = 1 - x
    return f"{x:.3f}"


keys = {
    "wtau": r"Weighted $\tau$",
    "evalfocus": r"Evaluation focus",
    # "tau": r"Standard $\tau$",
    # "clup": r"Average $p$-value",
}

outputs = [x for x in outputs if "data" in x]
outputs = [
    {
        "method": output["method_typst"],
        "method_ranker": output["method_ranker"],
        **{key: area_under_curve(output["data"], key) for key in keys.keys()},
    }
    for output in outputs
]
with open("../figures/simulation.json", "w") as f:
    json.dump(outputs, f, indent=2)
