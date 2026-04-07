# %%

from evaluation_bandit import algorithms, utils, estimators
import importlib
import random
import statistics

importlib.reload(utils)
importlib.reload(estimators)


def shuffled(ys):
    ys = list(ys)
    random.shuffle(ys)
    return ys


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])

for line in data:
    line["cost"] = 1

# %%
importlib.reload(algorithms)

budget = 200

data = shuffled(data)
# data.sort(key=lambda x: statistics.mean(x["scores"].values()))
model_scores_true = utils.items_to_model_scores(data)


def eval_both(model_scores):
    wtau = utils.wtau(estimators.mean(model_scores), estimators.mean(model_scores_true))
    print(f"{wtau:.3f}")


model_scores = algorithms.uniform(data, [budget])[0]
eval_both(model_scores)
model_scores = algorithms.greedy_oracle_invariant(
    data, [budget], batch_size=10, shuffle_repetitions=10
)[0]
eval_both(model_scores)
model_scores = algorithms.weighted_sampling(data, [budget])[0]
eval_both(model_scores)
model_scores = algorithms.confusion_minimization(data, budgets=[budget])[0]
eval_both(model_scores)

# %%
import numpy as np
import collections
import importlib

importlib.reload(algorithms)

BUDGETS = np.linspace(0.1, 1.0, 20, dtype=float)
budgets = [int(p * len(data) * len(data[0]["scores"])) for p in BUDGETS]
results = collections.defaultdict(list)

model_scores_true = utils.items_to_model_scores(data)


def eval_both(model_scores):
    wtau = utils.wtau(estimators.mean(model_scores), estimators.mean(model_scores_true))
    print(f"{wtau:.3f}")


for _ in range(5):
    data = shuffled(data)
    results["uniform"] += algorithms.uniform(data, budgets=budgets)
    results["greedy_oracle"] += algorithms.greedy_oracle(
        data,
        budgets=budgets,
        batch_size=25,
        batch_size_lookahead=75,
    )
    results["weighted_sampling"] += algorithms.weighted_sampling(data, budgets=budgets)
    results["confusion_minimization"] += algorithms.confusion_minimization(
        data,
        budgets=budgets,
        coldstart=10,
    )

for name, values in results.items():
    wtaus = [
        utils.wtau(estimators.mean(model_scores), estimators.mean(model_scores_true))
        for model_scores in values
    ]
    print(f"{name}: {statistics.mean(wtaus):.3f}")
