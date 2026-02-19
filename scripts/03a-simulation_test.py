# %%

from evaluation_bandit import algorithms, utils
import math
import importlib
import random


def shuffled(ys):
    ys = list(ys)
    random.shuffle(ys)
    return ys


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])
print(len(data))

# %%
importlib.reload(algorithms)
importlib.reload(utils)

budget = 400
print("                     ef   tau")


model_scores_true = {
    model: [item["scores"][model] for item in data] for model in data[0]["scores"]
}


def eval_both(model_scores):
    return (
        f"{utils.evalfocus(model_scores, model_scores_true, budget=budget):.3f}",
        f"{utils.wtau(model_scores, model_scores_true):.3f}",
    )


model_scores = [
    algorithms.uniform_nonsquare(shuffled(data), [budget])[0] for _ in range(10)
]
print(utils.stability(model_scores))

model_scores = [
    algorithms.upper_confidence_bound(shuffled(data), [budget])[0] for _ in range(10)
]
print(utils.stability(model_scores))
# print("uniform           ", *eval_both(model_scores))
# model_scores = algorithms.weighted_sampling_oracle(
#     data,
#     sampling_fn=lambda x, rank, total: 1 / (rank + 1),
#     budgets=[budget],
# )[0]
# print("rank oracle       ", *eval_both(model_scores))
# model_scores = algorithms.weighted_sampling(
#     data,
#     sampling_fn=lambda x, rank, total: 1 / (rank + 1),
#     budgets=[budget],
# )[0]
# print("rank              ", *eval_both(model_scores))
