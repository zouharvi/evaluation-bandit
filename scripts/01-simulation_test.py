# %%

from evaluation_bandit import algorithms, utils, estimators, simulation
import importlib
import random
import statistics


def shuffled(ys):
    ys = list(ys)
    random.shuffle(ys)
    return ys


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])
print(len(data))

for line in data:
    line["cost"] = 1

# %%
importlib.reload(algorithms)
importlib.reload(utils)
importlib.reload(estimators)

budget = 100

data = shuffled(data)
# data.sort(key=lambda x: statistics.mean(x["scores"].values()))
model_scores_true = utils.items_to_model_scores(data)


def eval_both(model_scores):
    wtau = utils.wtau(estimators.mean(model_scores), estimators.mean(model_scores_true))
    print(f"{wtau:.3f}")


model_scores = algorithms.uniform(data, [budget])[0]
eval_both(model_scores)
model_scores = algorithms.greedy_oracle(data, [budget], batch_size=10)[0]
eval_both(model_scores)
model_scores = algorithms.greedy_oracle_invariant(data, [budget], batch_size=10)[0]
eval_both(model_scores)


# %%
import matplotlib.pyplot as plt

data = simulation.subset2evaluate_to_sorter(
    method="metric_avg", metric="MetricX-25", cost_normalize=True
)(utils.load_data()[("wmt25", "cs-de_DE")])

plt.hist([x["subset2evaluate_utility"] for x in data])
plt.show()
plt.hist([x["cost"] for x in data])
plt.show()
