# %%

from evaluation_bandit import algorithms, utils
import math
import importlib


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])
print(len(data))

# %%
importlib.reload(algorithms)
importlib.reload(utils)

budget = 500
print("                     ef   tau")


def eval_both(model_scores):
    return (
        f"{utils.evalfocus(model_scores, model_scores_true, budget=budget):.3f}",
        f"{utils.wtau(model_scores, model_scores_true):.3f}",
    )


model_scores_true = {
    model: [item["scores"][model] for item in data] for model in data[0]["scores"]
}
model_scores = algorithms.uniform_nonsquare(data, budget)
print("uniform          ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling_oracle(
    data,
    sampling_fn=lambda x, rank, total: 1 / (rank + 1),
    budgets=[budget],
)[0]
print("rank oracle      ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling(
    data,
    sampling_fn=lambda x, rank, total: 1 / (rank + 1),
    budgets=[budget],
)[0]
print("rank             ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling_oracle(
    data,
    sampling_fn=lambda x, rank, total: 1 / math.sqrt(rank + 1),
    budgets=[budget],
)[0]
# print("rank-sqrt oracle ", *eval_both(model_scores))
# model_scores = algorithms.weighted_sampling(
#     data,
#     sampling_fn=lambda x, rank, total: 1 / math.sqrt(rank + 1),
#     budgets=[budget],
# )[0]
# print("rank-sqrt        ", *eval_both(model_scores))
