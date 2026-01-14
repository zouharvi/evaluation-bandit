# %%

from translation_bandit import simulation, algorithms
SEEDS = 100

outputs_baseline = simulation.simulate(
    fn=algorithms.baseline,
    seeds=SEEDS ,
)
outputs_successive_rejects_constant = simulation.simulate(
    fn=algorithms.successive_rejects,
    fn_kwargs=dict(phases="constant"),
    seeds=SEEDS,
)
# outputs_successive_rejects_prioritize_all = simulation.simulate(
#     fn=algorithms.successive_rejects,
#     fn_kwargs=dict(phases="prioritize_all"),
#     seeds=SEEDS,
# )
# outputs_successive_rejects_prioritize_top = simulation.simulate(
#     fn=algorithms.successive_rejects,
#     fn_kwargs=dict(phases="prioritize_top"),
#     seeds=SEEDS,
# )
# def epsilon_fn_top(rank, total):
#     return 1 if rank <= 5 else 0.1
# outputs_epsilon_greedy_top = simulation.simulate(
#     fn=algorithms.epsilon_greedy,
#     fn_kwargs=dict(epsilon=epsilon_fn_top),
#     accepts_budgets=True,
#     seeds=SEEDS,
# )
def epsilon_fn_smooth(rank, total):
    return 1 / (rank + 1)
outputs_epsilon_greedy_smooth = simulation.simulate(
    fn=algorithms.epsilon_greedy,
    fn_kwargs=dict(epsilon=epsilon_fn_smooth),
    accepts_budgets=True,
    seeds=SEEDS,
)
outputs_confidence_ambiguity_1x1 = simulation.simulate(
    fn=algorithms.confidence_ambiguity,
    fn_kwargs=dict(weight_ci_p=(1, 1)),
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
        )
        if color == "black" or True:
            ax.fill_between(
                xs,
                [np.mean([x[key+"_ci"][0] for x in xs]) for xs in data_by_budget],
                [np.mean([x[key+"_ci"][1] for x in xs]) for xs in data_by_budget],
                alpha=0.4,
                color=color,
                linewidth=0.0,
            )

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 4.5))
axs = axs.flatten()

plot_output(outputs_baseline, "Random", axs, color="black")
plot_output(outputs_successive_rejects_constant, "Successive rejects", axs, color="C0")
# plot_output(outputs_successive_rejects_prioritize_all, "Successive Rejects (prioritize all)", axs)
# plot_output(outputs_successive_rejects_prioritize_top, "Successive Rejects (prioritize top)", axs)
# plot_output(outputs_epsilon_greedy_top, "Epsilon-Greedy (top)", axs)
plot_output(outputs_epsilon_greedy_smooth, r"$\epsilon$-Greedy", axs, color="C1")
plot_output(outputs_confidence_ambiguity_1x1, r"Minimize overlap", axs, color="C2")

for ax in axs:
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%')) # type: ignore
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.2f}')) # type: ignore
    ax.set_xlabel("Budget proportion")


def format_ax_label(ax, x, y, text):
    # \uparrow in top left of axis
    ax.text(
        x, y,
        text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
    )

format_ax_label(axs[0], 0.02, 0.90, r"Standard $\tau$ $\uparrow$")
format_ax_label(axs[1], 0.02, 0.90, r"Weighted $\tau$ $\uparrow$")
format_ax_label(axs[2], 0.45, 0.90, r"Average $p$-value $\downarrow$")
format_ax_label(axs[3], 0.45, 0.20, r"Evaluation focus $\uparrow$")

axs[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%')) # type: ignore
axs[0].set_ylim(0.85, 1.0+0.01)
axs[1].set_ylim(0.85, 1.0+0.01)
axs[2].set_ylim(None, 0.6)

plt.tight_layout(pad=0)
plt.subplots_adjust(hspace=0.3)
plt.savefig("../figures/simulation_wmt25.svg")
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(8, 0.2))
handles, labels = axs[0].get_legend_handles_labels()
fig_legend.legend(handles, labels, loc='center', ncol=4, frameon=False)
fig_legend.tight_layout(pad=0)
plt.axis('off')
plt.savefig("../figures/simulation_wmt25_legend.svg")
plt.show()

# %%
import os
import pickle
os.makedirs("cache/", exist_ok=True)
with open("cache/outputs_confidence_ambiguity_1x1.pkl", "wb") as f:
    pickle.dump(outputs_confidence_ambiguity_1x1, f)