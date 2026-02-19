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
        "method_typst": "Sampling rank (oracle)",
        "method_latex": None,
        "method": "weighted_sampling_oracle_rank",
    },
    {
        "method_typst": "Sampling rank-sqrt",
        "method_latex": None,
        "method": "weighted_sampling_ranksqrt",
    },
    {
        "method_typst": "Sampling rank$#none^2$",
        "method_latex": None,
        "method": "weighted_sampling_rankpow2",
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
    # only for extra
    {
        "method_typst": "Ambiguity reduction $lambda$=$1$",
        "method_latex": None,
        "method": "ambiguity_reduction_11",
    },
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
    {
        "method_typst": "Thompson sampling",
        "method_latex": None,
        "method": "thompson_sampling",
    },
    {
        "method_typst": "$p$-value rejects",
        "method_latex": None,
        "method": "pvalue_rejects",
    },
    {
        "method_typst": "Successive halving",
        "method_latex": None,
        "method": "successive_halving",
    },
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
        # "cometconfidence",
        # "sentinel_mqm",
        # "precomet_diffdisc",
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
import scipy.signal
import matplotlib.pyplot as plt
import importlib

importlib.reload(utils_fig)


def smooth(ys):
    return scipy.signal.savgol_filter(ys, 7, 2)


def plot_output(outputs, label, axs, color=None):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    print(label)
    xs = [xs[0]["budget"] for xs in data_by_budget]
    for ax, key in zip(
        axs,
        ["wtau", "stability", "evalfocus"],
    ):
        ax.plot(
            xs,
            smooth([np.mean([x[key] for x in xs]) for xs in data_by_budget]),
            label=label,
            color=color,
            linewidth=2.0,
            zorder=2 if label == "Uniform" else 1,
        )
        if key != "stability":
            ax.fill_between(
                xs,
                smooth(
                    [np.mean([x[key + "_ci"][0] for x in xs]) for xs in data_by_budget]
                ),
                smooth(
                    [np.mean([x[key + "_ci"][1] for x in xs]) for xs in data_by_budget]
                ),
                alpha=0.4,
                color=color,
                linewidth=0.0,
            )


fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 2.4))
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
axs[0].set_ylim(0.88, None)
axs[0].set_ylabel(r"Ranking ($\tau_\omega$)", labelpad=-5)
axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
axs[1].set_ylim(0.88, None)
axs[1].set_ylabel(r"Stability ($\tau_\omega$)", labelpad=-5)
axs[2].set_ylim(50, None)
axs[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))  # type: ignore
axs[2].set_ylabel("\nEvaluation focus", labelpad=1)

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation.svg")
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(8, 0.4))
handles, labels = axs[0].get_legend_handles_labels()
# make the lines three times as thick
for handle in handles:
    handle.set_linewidth(6.0)

fig_legend.legend(
    handles,
    labels,
    loc="center",
    ncol=3,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=1,
)
fig_legend.tight_layout(pad=0)
plt.axis("off")
plt.savefig("../figures/simulation_legend.svg")
plt.show()


# area under curve table


def area_under_curve(outputs, key):
    if key not in outputs[0]:
        return None
    x = np.mean([x[key] for x in outputs])
    if key == "evalfocus":
        return f"{x:.1f}"
    else:
        return f"{x:.3f}"
    # data_by_budget = collections.defaultdict(list)
    # for output in outputs:
    #     data_by_budget[output["budget"]].append(output)
    # data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    # x = np.trapezoid(
    #     y=[
    #         np.mean([x[key] for x in xs])
    #         for xs in data_by_budget
    #         if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 1.0
    #     ],
    #     x=[
    #         xs[0]["budget"]
    #         for xs in data_by_budget
    #         if xs[0]["budget"] >= 0.1 and xs[0]["budget"] <= 1.0
    #     ],
    # )
    # return f"{x / (1 - 0.1):.3f}"


outputs = [x for x in outputs if "data" in x]
outputs = [
    {
        "method": output["method"],
        "method_typst": output["method_typst"],
        "method_ranker": output["method_ranker"],
        **{
            key: area_under_curve(output["data"], key)
            for key in ["wtau", "evalfocus", "tau", "avg_pval", "stability"]
        },
    }
    for output in outputs
]
with open("../figures/simulation.json", "w") as f:
    json.dump(outputs, f, indent=2)
