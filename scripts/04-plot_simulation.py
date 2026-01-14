# %%

from translation_bandit import simulation, algorithms
SEEDS = 10

outputs_baseline = simulation.simulate(
    fn=algorithms.baseline,
    seeds=SEEDS ,
)
outputs_successive_rejects_constant = simulation.simulate(
    fn=algorithms.successive_rejects,
    fn_kwargs=dict(phases="constant"),
    seeds=SEEDS,
)
outputs_successive_rejects_prioritize_all = simulation.simulate(
    fn=algorithms.successive_rejects,
    fn_kwargs=dict(phases="prioritize_all"),
    seeds=SEEDS,
)
outputs_successive_rejects_prioritize_top = simulation.simulate(
    fn=algorithms.successive_rejects,
    fn_kwargs=dict(phases="prioritize_top"),
    seeds=SEEDS,
)
def epsilon_fn_top5(rank, total):
    return 1 if rank <= 5 else 0.1
outputs_epsilon_greedy_top5 = simulation.simulate(
    fn=algorithms.epsilon_greedy,
    fn_kwargs=dict(epsilon=epsilon_fn_top5),
    accepts_budgets=True,
    seeds=SEEDS,
)
def epsilon_fn_smooth(rank, total):
    return 1 / (rank + 1)
outputs_epsilon_greedy_smooth = simulation.simulate(
    fn=algorithms.epsilon_greedy,
    fn_kwargs=dict(epsilon=epsilon_fn_smooth),
    accepts_budgets=True,
    seeds=SEEDS,
)

outputs_confidence_ambiguity_rank_11 = simulation.simulate(
    fn=algorithms.confidence_ambiguity_rank,
    fn_kwargs=dict(weight_ci_p=(1, 1)),
    accepts_budgets=True,
    seeds=SEEDS,
)
outputs_confidence_ambiguity_rank_01 = simulation.simulate(
    fn=algorithms.confidence_ambiguity_rank,
    fn_kwargs=dict(weight_ci_p=(0, 1)),
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
    axs[0].plot(
        xs,
        [np.mean([x["tau"] for x in xs]) for xs in data_by_budget],
        label=label,
        color=color,
        linewidth=2.0,
    )
    axs[1].plot(
        xs,
        [np.mean([x["wtau"] for x in xs]) for xs in data_by_budget],
        label=label,
        color=color,
        linewidth=2.0,
    )
    axs[2].plot(
        xs,
        [np.mean([x["clup"] for x in xs]) for xs in data_by_budget],
        label=label,
        color=color,
        linewidth=2.0,
    )

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 3.5))

plot_output(outputs_baseline, "Baseline", axs, color="black")
plot_output(outputs_successive_rejects_constant, "Successive Rejects (constant)", axs)
# plot_output(outputs_successive_rejects_prioritize_all, "Successive Rejects (prioritize all)", axs)
# plot_output(outputs_successive_rejects_prioritize_top, "Successive Rejects (prioritize top)", axs)
plot_output(outputs_epsilon_greedy_top5, "Epsilon-Greedy (top 5)", axs)
plot_output(outputs_epsilon_greedy_smooth, "Epsilon-Greedy (smooth)", axs)
plot_output(outputs_confidence_ambiguity_rank_11, "Cluster-Max CI-Min", axs)
plot_output(outputs_confidence_ambiguity_rank_01, "Cluster-Max", axs)

axs[0].set_ylabel("Tau")
axs[1].set_ylabel("Weighted Tau")
axs[2].set_ylabel("Average $p$-value")
for ax in axs:
    ax.set_xlabel("Budget proportion")
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x * 100)}%')) # type: ignore
plt.tight_layout(pad=0)
plt.show()


# plot only legend
fig_legend = plt.figure(figsize=(4, 1))
handles, labels = axs[0].get_legend_handles_labels()
fig_legend.legend(handles, labels, loc='center', ncol=2, frameon=False)
fig_legend.tight_layout(pad=0)
plt.axis('off')
plt.show()
