# %%

from translation_tournament import simulation, algorithms

outputs_baseline = simulation.simulate(
    fn=algorithms.baseline
)
outputs_successive_rejects_constant = simulation.simulate(
    fn=lambda data, budget: algorithms.successive_rejects(data, budget, phases="constant")
)
outputs_successive_rejects_prioritize_all = simulation.simulate(
    fn=lambda data, budget: algorithms.successive_rejects(data, budget, phases="prioritize_all")
)
outputs_successive_rejects_prioritize_top = simulation.simulate(
    fn=lambda data, budget: algorithms.successive_rejects(data, budget, phases="prioritize_top")
)
outputs_epsilon_greedy_top5 = simulation.simulate(
    fn=lambda data, budget: algorithms.epsilon_greedy(data, budget, epsilon=lambda rank, total: 1 if rank <=5 else 0.1)
)
outputs_epsilon_greedy_smooth = simulation.simulate(
    fn=lambda data, budget: algorithms.epsilon_greedy(data, budget, epsilon=lambda rank, total: 1/(rank + 1))
)
outputs_confidence_ambiguity_rank_11 = simulation.simulate(
    fn=lambda data, budgets: algorithms.confidence_ambiguity_rank(data, budgets=budgets, weight_ci_p=(1, 1)),
    accepts_budgets=True,
)
# outputs_confidence_ambiguity_rank_10 = simulation.simulate(
#     fn=lambda data, budgets: algorithms.confidence_ambiguity_rank(data, budgets=budgets, weight_ci_p=(1, 0)),
#     accepts_budgets=True,
# )
# outputs_confidence_ambiguity_rank_01 = simulation.simulate(
#     fn=lambda data, budgets: algorithms.confidence_ambiguity_rank(data, budgets=budgets, weight_ci_p=(0, 1)),
#     accepts_budgets=True,
# )

# %%

# TODO: multiple runs due to stochasticity

# %%

from translation_tournament import utils_fig
import collections
import numpy as np
import matplotlib.pyplot as plt

def safe_mean(ys):
    return np.mean([y for y in ys if y is not None and not np.isnan(y)])

def plot_output(outputs, label, axs):
    data_by_name = collections.defaultdict(list)
    for output in outputs:
        data_by_name[output["data_name"]].append(output)
    data_by_name = list(data_by_name.values())

    # TODO: change XSs to be proportions
    xs = list(range(len(data_by_name[0])))
    axs[0].plot(
        xs,
        [safe_mean([data[i]["tau"] for data in data_by_name]) for i in xs],
        label=label,
    )
    axs[1].plot(
        xs,
        [safe_mean([data[i]["wtau"] for data in data_by_name]) for i in xs],
        label=label,
    )
    axs[2].plot(
        xs,
        [safe_mean([data[i]["clup"] for data in data_by_name]) for i in xs],
        label=label,
    )

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(8, 3.5))

plot_output(outputs_baseline, "Baseline", axs)
plot_output(outputs_successive_rejects_constant, "Successive Rejects (constant)", axs)
plot_output(outputs_successive_rejects_prioritize_all, "Successive Rejects (prioritize all)", axs)
plot_output(outputs_successive_rejects_prioritize_top, "Successive Rejects (prioritize top)", axs)
plot_output(outputs_epsilon_greedy_top5, "Epsilon-Greedy (top 5)", axs)
plot_output(outputs_epsilon_greedy_smooth, "Epsilon-Greedy (smooth)", axs)
plot_output(outputs_confidence_ambiguity_rank_11, "Conf-Amb Rank (1,1)", axs)
# plot_output(outputs_confidence_ambiguity_rank_10, "Conf-Amb Rank (1,0)", axs)
# plot_output(outputs_confidence_ambiguity_rank_01, "Conf-Amb Rank (0,1)", axs)

axs[0].set_ylabel("Tau")
axs[1].set_ylabel("Weighted Tau")
axs[2].set_ylabel("Average $p$-value")
for ax in axs:
    ax.set_xlabel("Budget proportion")
axs[2].legend()
plt.tight_layout(pad=0)
plt.show()
