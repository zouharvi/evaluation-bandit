# %%

import json
import tqdm


def read_computed(method, method_ranker, method_estimator, method_estimator_eval):
    with open(
        f"../computed/02/{method}#{method_ranker}#{method_estimator}#{method_estimator_eval}.json",
        "r",
    ) as f:
        return json.load(f)


outputs = [
    {
        "method_typst": "Uniform",
        "method_latex": "Uniform",
        "method": "uniform",
    },
    {
        "method_typst": "Uniform (sampling)",
        "method_latex": None,
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
    # {
    #     "method_typst": "Sampling $\"rank\"^1$ (oracle)",
    #     "method_latex": None,
    #     "method": "weighted_sampling_oracle_rank",
    #     "basiconly": True,
    # },
    {
        "method_typst": 'Sampling $"rank"^2$',
        "method_latex": None,
        "method": "weighted_sampling_rankpow2",
    },
    {
        "method_typst": 'Sampling $root(2, "rank")$',
        "method_latex": None,
        "method": "weighted_sampling_rankpow0.5",
        "basiconly": True,
    },
    {
        "method_typst": 'Sampling $root(4, "rank")$',
        "method_latex": None,
        "method": "weighted_sampling_rankpow0.25",
        "basiconly": True,
    },
    {
        "method_typst": 'Sampling rank $sqrt("top 3")$',
        "method_latex": None,
        "method": "weighted_sampling_ranktop3sqrt",
        "basiconly": True,
    },
    {
        "method_typst": 'Sampling $"rank"^1$ (rev)',
        "method_latex": None,
        "method": "weighted_sampling_rankrevpow1",
        "basiconly": True,
    },
    {
        "method_typst": "Sampling $epsilon$-greedy",
        "method_latex": None,
        "method": "weighted_sampling_epsilongreedy",
    },
    {
        "method_typst": "Sampling Bolzmann",
        "method_latex": None,
        "method": "weighted_sampling_bolzmann",
    },
    {
        "method_typst": "Upper confidence bound",
        "method_latex": "Upper confidence bound",
        "method": "ucb",
    },
    {
        "method_typst": "Confusion minimization",
        "method_latex": "Confusion minimization",
        "method": "confusion_minimization",
    },
    {
        "method_typst": "Greedy oracle",
        "method_latex": "Greedy oracle",
        "method": "greedy_oracle_invariant_wtau_pow2",
    },
    {
        "method_typst": 'Greedy oracle $"rank"^1$',
        "method_latex": None,
        "method": "greedy_oracle_invariant_wtau_pow1",
        "basiconly": True,
    },
    {
        "method_typst": 'Greedy oracle $sqrt("rank")$',
        "method_latex": None,
        "method": "greedy_oracle_invariant_wtau_pow05",
        "basiconly": True,
    },
    # {
    #     "method_typst": "Greedy oracle $root(4, \"rank\")$",
    #     "method_latex": None,
    #     "method": "greedy_oracle_invariant_wtau_pow025",
    #     "basiconly": True,
    # },
    {
        "method_typst": 'Greedy oracle rank $"top" 3$',
        "method_latex": None,
        "method": "greedy_oracle_invariant_wtau_top3",
        "basiconly": True,
    },
    {
        "method_typst": 'Greedy oracle $"rank"^1$ (rev)',
        "method_latex": None,
        "method": "greedy_oracle_invariant_wtau_revpow1",
        "basiconly": True,
    },
    {
        "method_typst": "Ambiguity reduction $lambda$=$1$",
        "method_latex": None,
        "method": "ambiguity_reduction_11",
        "basiconly": True,
    },
    {
        "method_typst": "Ambiguity reduction $lambda$=$0$",
        "method_latex": None,
        "method": "ambiguity_reduction_01",
        "basiconly": True,
    },
    {
        "method_typst": "Ambiguity reduction $lambda$=$infinity$",
        "method_latex": None,
        "method": "ambiguity_reduction_10",
        "basiconly": True,
    },
    {
        "method_typst": "Thompson sampling",
        "method_latex": None,
        "method": "thompson_sampling",
        "basiconly": True,
    },
    {
        "method_typst": "$p$-value rejects",
        "method_latex": None,
        "method": "pvalue_rejects",
        "basiconly": True,
    },
    {
        "method_typst": "Successive halving",
        "method_latex": None,
        "method": "successive_halving",
        "basiconly": True,
    },
]

outputs = [
    output
    | {
        "method_ranker": method_ranker,
        "method_estimator": method_estimator,
        "method_estimator_eval": method_estimator,
    }
    for output in outputs
    for method_ranker in [
        "random",
        "metricavg",
        "metricavg_cost",
        "rev_metricavg",
    ]
    for method_estimator in ["additive", "mean"]
]

for output in tqdm.tqdm(outputs):
    if output.get("basiconly", False) and (
        output["method_ranker"] != "random"
        or output["method_estimator"] != "mean"
        or output["method_estimator_eval"] != "mean"
    ):
        continue
    try:
        output["outputs"] = read_computed(
            output["method"],
            output["method_ranker"],
            output["method_estimator"],
            output["method_estimator_eval"],
        )
    except FileNotFoundError:
        print(
            f"Warning: computed file for method {output['method']}#{output['method_ranker']}#{output['method_estimator']}#{output['method_estimator_eval']} not found."
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


outputs_greedy_oracle = [
    output["outputs"]
    for output in outputs
    if output["method"] == "greedy_oracle_invariant_wtau_pow2"
    and output["method_ranker"] == "random"
    and output["method_estimator"] == "mean"
    and output["method_estimator_eval"] == "mean"
][0]


def plot_output(outputs, label, ax, color=None):
    data_by_budget = collections.defaultdict(list)
    for output in outputs:
        data_by_budget[output["budget"]].append(output)
    data_by_budget = sorted(data_by_budget.values(), key=lambda d: d[0]["budget"])

    xs = [xs[0]["budget"] for xs in data_by_budget]
    ax.plot(
        xs,
        smooth([np.mean([x["wtau"] for x in xs]) for xs in data_by_budget]),
        label=label + f" ({np.mean([x['wtau'] for x in outputs]):.3f})",
        color=color,
        linewidth=2.0,
        zorder=2 if label == "Uniform" else 1,
    )
    ax.fill_between(
        xs,
        smooth([np.mean([x["wtau_ci"][0] for x in xs]) for xs in data_by_budget]),
        smooth([np.mean([x["wtau_ci"][1] for x in xs]) for xs in data_by_budget]),
        alpha=0.3,
        color=color,
        linewidth=0.0,
    )


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2.4))
# axs = axs.flatten()

outputs_to_plot = [
    output
    for output in outputs
    if output["method_latex"] is not None
    and output["method_ranker"] == "random"
    and output["method_estimator"] == "mean"
    and output["method_estimator_eval"] == "mean"
    and "outputs" in output
]

for output_i, output in enumerate(outputs_to_plot):
    plot_output(
        output["outputs"],
        output["method_latex"],
        ax,
        color="black" if output["method"] == "uniform" else f"C{output_i - 1}",
    )

ax.spines[["top", "right"]].set_visible(False)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))  # type: ignore
ax.set_xlabel("Budget proportion")
ax.set_xlim(0.2, 1.0)


ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x * 100)}%"))  # type: ignore
ax.set_ylim(0.9, 1)
ax.set_ylabel(r"$\tau_\omega$", labelpad=-5)

ax.set_facecolor("none")
fig.patch.set_facecolor("none")

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation.svg")
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(2.5, 1.2))
handles, labels = ax.get_legend_handles_labels()
# make the lines three times as thick
for handle in handles:
    handle.set_linewidth(6.0)

fig_legend.legend(
    handles,
    labels,
    loc="center",
    ncol=1,
    frameon=False,
    handlelength=1,
    handletextpad=0.5,
    columnspacing=1,
)
fig_legend.tight_layout(pad=0)
plt.axis("off")
plt.savefig("../figures/simulation_legend.svg")
plt.show()


def area_under_curve(outputs, key):
    if key not in outputs[0]:
        return None
    x = np.mean([x[key] for x in outputs])
    if key in {"evalfocus", "avg_payoff"}:
        return f"{x:.1f}"
    else:
        return f"{x:.3f}"


OBJECTIVES = [
    "wtau",
    "tau",
    "stability",
    "evalfocus",
    "avg_pval",
    "avg_payoff",
    "wtau_pow1",
    "wtau_pow2",
    "wtau_pow05",
    "wtau_top3",
    "wtau_top1",
    "wtau_bot3",
    "wtau_middle3",
    "wtau_revpow1",
]

outputs = [x for x in outputs if "outputs" in x]
outputs_out = [
    {
        "method": output["method"],
        "method_typst": output["method_typst"],
        "method_ranker": output["method_ranker"],
        "method_estimator": output["method_estimator"],
        "method_estimator_eval": output["method_estimator_eval"],
        **{key: area_under_curve(output["outputs"], key) for key in OBJECTIVES},
    }
    for output in outputs
]


for item in outputs_out:
    # find greedy oracle additive and set to item_oracle_mean
    if (
        item["method"] == "greedy_oracle"
        and item["method_estimator_eval"] == "additive"
    ):
        item_super = [
            item
            for item in outputs_out
            if item["method"] == "greedy_oracle"
            and item["method_ranker"] == "random"
            and item["method_estimator"] == "mean"
            and item["method_estimator_eval"] == "mean"
        ][0]
        for key in OBJECTIVES:
            item[key] = item_super[key]

    # find confusion_minimization with additive method_estimator_eval
    if (
        item["method"] == "confusion_minimization"
        and item["method_estimator_eval"] == "additive"
    ):
        item_super = [
            item
            for item in outputs_out
            if item["method"] == "confusion_minimization"
            and item["method_ranker"] == item["method_ranker"]
            and item["method_estimator"] == "mean"
            and item["method_estimator_eval"] == "mean"
        ][0]
        for key in OBJECTIVES:
            item[key] = item_super[key]

with open("../figures/simulation.json", "w") as f:
    json.dump(outputs_out, f, indent=2)
