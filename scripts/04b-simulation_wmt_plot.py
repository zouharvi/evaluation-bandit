# %%

import json


def read_computed(method):
    with open(f"../computed/simulation_wmt_{method}.json", "r") as f:
        return json.load(f)


outputs = [
    {"typst": "Uniform", "latex": "Uniform", "internal": "uniform"},
    {
        "typst": "Successive rejects",
        "latex": "Successive rejects",
        "internal": "successive_rejects_constant",
    },
    {
        "typst": "Sampling rank-based",
        "latex": "Sampling rank-based",
        "internal": "stochastic_sampling_ranksmooth",
    },
    {
        "typst": "Sampling $epsilon$-Greedy",
        "latex": "Sampling $\\epsilon$-Greedy",
        "internal": "stochastic_sampling_epsilongreedy",
    },
    {
        "typst": "Sampling Bolzmann",
        "latex": "Sampling Bolzmann",
        "internal": "stochastic_sampling_bolzmann",
    },
    {
        "typst": "Upper Confidence Bound",
        "latex": "Upper Confidence Bound",
        "internal": "ucb_c50",
    },
    {
        "typst": "Ambiguity reduction $lambda$=$1$",
        "latex": "Ambiguity reduction $\\lambda=1$",
        "internal": "ambiguity_reduction_11",
    },
    # no LaTeX for s2e
    {
        "typst": "Ambiguity reduction $lambda$=$0$",
        "latex": None,
        "internal": "ambiguity_reduction_01",
    },
    {
        "typst": "Ambiguity reduction $lambda$=$infinity$",
        "latex": None,
        "internal": "ambiguity_reduction_10",
    },
    # {"typst": "UCB c=100", "latex": None, "internal": "ucb_c100"},
    # {"typst": "UCB c=200", "latex": None, "internal": "ucb_c200"},
    {"typst": "MetricVar", "latex": None, "internal": "s2e_metricvar"},
    {"typst": "MetricAvg", "latex": None, "internal": "s2e_metricavg"},
    {"typst": "MetricCons", "latex": None, "internal": "s2e_metriccons"},
    {"typst": "$k$-means", "latex": None, "internal": "s2e_kmeans"},
    {"typst": "DiffUse", "latex": None, "internal": "s2e_diffuse"},
    {"typst": "Brute Greedy", "latex": None, "internal": "s2e_brute_greedy"},
    {"typst": "Brute", "latex": None, "internal": "s2e_brute"},
    {"typst": "Diversity BLEU", "latex": None, "internal": "s2e_diversity_bleu"},
    {"typst": "Diversity Unigram", "latex": None, "internal": "s2e_diversity_unigram"},
    {"typst": "Diversity LM", "latex": None, "internal": "s2e_diversity_lm"},
    {"typst": "Instant confidence", "latex": None, "internal": "s2e_cometconfidence"},
    {"typst": "Sentinel MQM", "latex": None, "internal": "s2e_sentinel_mqm"},
    {"typst": "Pre-Comet DiffDisc", "latex": None, "internal": "s2e_precomet_diffdisc"},
]

for output in outputs:
    try:
        output["data"] = read_computed(output["internal"])
    except FileNotFoundError:
        print(f"Warning: computed file for method {output['internal']} not found.")


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
        ["tau", "wtau_smooth", "clup", "evalcount_smooth"],
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


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 4.5))
axs = axs.flatten()

for output_i, output in enumerate(outputs):
    if output["latex"] is None or "data" not in output:
        continue
    plot_output(
        output["data"],
        output["latex"],
        axs,
        color="black" if output["internal"] == "uniform" else f"C{output_i - 1}",
    )

for ax in axs:
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))  # type: ignore
    ax.set_xlabel("Budget proportion")
    ax.set_xlim(0.1, 0.9)


def format_ax_label(ax, x, y, text):
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
format_ax_label(axs[2], 0.52, 0.90, r"Average $p$-value $\downarrow$")
format_ax_label(axs[3], 0.52, 0.20, r"Evaluation focus $\uparrow$")

axs[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
axs[0].set_ylim(0.85, 1.0 + 0.01)
axs[1].set_ylim(0.85, 1.0 + 0.01)
axs[2].set_ylim(None, 0.55)
axs[3].set_ylim(0.25, None)

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation_wmt.svg")
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
plt.savefig("../figures/simulation_wmt_legend.svg")
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
    ) / (0.9 - 0.1)
    if key == "clup":
        x = 1 - x
    return f"{x:.3f}"


keys = {
    "tau": r"Standard $\tau$",
    "wtau_smooth": r"Weighted $\tau$",
    "clup": r"Average $p$-value",
    "evalcount_smooth": r"Evaluation focus",
}

outputs = [x for x in outputs if "data" in x]

output = outputs[0]
print(
    "[]",
    f"[{output['typst']:<37}]",
    *(
        f'cell{cell_i + 1}("{area_under_curve(output["data"], key)}")'
        for cell_i, key in enumerate(keys.keys())
    ),
    sep=", ",
    end=",\n",
)
print("cmidrule(end: 2),")

outputs_local = [
    x
    for x in outputs
    if not x["internal"].startswith("s2e_") and x["internal"] != "uniform"
]
for output_i, output in enumerate(outputs_local):
    if output_i == 0:
        print(
            f"""
table.cell(
    rowspan: {len(outputs_local)}, 
    align: center,
    rotate(-90deg, reflow: true)[*Model-selection*]
),
""",
            end="",
        )
    print(
        f"[{output['typst']:<37}]",
        *(
            f'cell{cell_i + 1}("{area_under_curve(output["data"], key)}")'
            for cell_i, key in enumerate(keys.keys())
        ),
        sep=", ",
        end=",\n",
    )


print("cmidrule(end: 2),")
outputs_local = [
    x
    for x in outputs
    if x["internal"].startswith("s2e_") and x["internal"] != "baseline"
]
for output_i, output in enumerate(outputs_local):
    if output_i == 0:
        print(
            f"""
table.cell(
    rowspan: {len(outputs_local)}, 
    align: center,
    rotate(-90deg, reflow: true)[*Item-selection*]
),
""",
            end="",
        )
    print(
        f"[{output['typst']:<37}]",
        *(
            f'cell{cell_i + 1}("{area_under_curve(output["data"], key)}")'
            for cell_i, key in enumerate(keys.keys())
        ),
        sep=", ",
        end=",\n",
    )
