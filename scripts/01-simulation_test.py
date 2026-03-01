# %%

from evaluation_bandit import algorithms, utils, estimators, simulation
import importlib
import random
import statistics
import math


def shuffled(ys):
    ys = list(ys)
    random.shuffle(ys)
    return ys


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])
print(len(data))

# %%
importlib.reload(algorithms)
importlib.reload(utils)
importlib.reload(estimators)

budget = 400
print("                     ef   tau")

data = shuffled(data)
data.sort(key=lambda x: statistics.mean(x["scores"].values()))

model_scores_true = {
    model: [item["scores"][model] for item in data] for model in data[0]["scores"]
}


def eval_both(model_scores):
    return (
        f"{utils.evalfocus(model_scores, model_scores_true, budget=budget):.3f}",
        f"{utils.wtau(model_scores, model_scores_true):.3f}",
    )


model_scores = algorithms.uniform_nonsquare((data), [budget])[0]
print(model_scores)
print(utils.wtau(estimators.mean(model_scores), estimators.mean(model_scores_true)))


# %%
import matplotlib.pyplot as plt

data = simulation.subset2evaluate_to_sorter(
    method="metric_avg", metric="MetricX-25", cost_normalize=True
)(utils.load_data()[("wmt25", "cs-de_DE")])

plt.hist([x["subset2evaluate_utility"] for x in data])
plt.show()
plt.hist([x["cost"] for x in data])
plt.show()
