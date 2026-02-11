# %%

from evaluation_bandit import algorithms, utils
import math
import importlib


data = utils.data_humanscores_only(utils.load_data()[("wmt25", "cs-de_DE")])
print(len(data))

# %%
importlib.reload(algorithms)
importlib.reload(utils)

budget = 400
print("                     ef   tau")


def eval_both(model_scores):
    return (
        f"{utils.evalfocus(model_scores, model_scores_true, budget=budget):.3f}",
        f"{utils.wtau(model_scores, model_scores_true):.3f}",
    )


model_scores_true = {
    model: [item["scores"][model] for item in data] for model in data[0]["scores"]
}
model_scores = algorithms.uniform_nonsquare(data, [budget])[0]
print("uniform           ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling_oracle(
    data,
    sampling_fn=lambda x, rank, total: 1 / (rank + 1),
    budgets=[budget],
)[0]
print("rank oracle       ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling(
    data,
    sampling_fn=lambda x, rank, total: 1 / (rank + 1),
    budgets=[budget],
)[0]
print("rank              ", *eval_both(model_scores))
model_scores = algorithms.weighted_sampling_oracle(
    data,
    sampling_fn=lambda x, rank, total: 1 / math.sqrt(rank + 1),
    budgets=[budget],
)[0]
model_scores = algorithms.pvalue_rejects(
    data,
    [budget],
    threshold=0.05,
)[0]
print("p-value rejects   ", *eval_both(model_scores))
model_scores = algorithms.successive_halving(data, budget)
print("successive halving", *eval_both(model_scores))
# print("rank-sqrt oracle ", *eval_both(model_scores))
#     sampling_fn=lambda x, rank, total: 1 / math.sqrt(rank + 1),
#     budgets=[budget],
# )[0]
# print("rank-sqrt        ", *eval_both(model_scores))

model_scores = algorithms.upper_confidence_bound(data, [budget], variant="ucb1")[0]
print("ucb1              ", *eval_both(model_scores))
model_scores = algorithms.upper_confidence_bound(data, [budget], variant="lilucb")[0]
print("lilucb            ", *eval_both(model_scores))

model_scores = algorithms.thompson_sampling(data, [budget])[0]
print("thompson sampling ", *eval_both(model_scores))
