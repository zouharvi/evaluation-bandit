# %%

from evaluation_bandit import algorithms, utils
import importlib

importlib.reload(algorithms)
importlib.reload(utils)
import numpy as np

# ['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-ko_KR', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA', 'en-zh_CN', 'ja-zh_CN'])

data = utils.load_data_single(langs="en-cs_CZ")
for item in data:
    for i in range(0):
        item["scores"] |= {
            model + "*" * i: score + np.random.normal(0, 0.001)
            for model, score in item["scores"].items()
        }

model_scores_true = {
    model: [item["scores"][model] for item in data] for model in data[0]["scores"]
}


def budgets():
    return np.linspace(200, int(len(data) * len(data[0]["scores"]) * 0.5), 5, dtype=int)


for budget in budgets():
    model_scores = algorithms.uniform(data, budget)
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )

print()
for budget in budgets():
    model_scores = algorithms.successive_rejects(data, budget, _phases="constant")
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )

print()
for budget in budgets():
    model_scores = algorithms.successive_rejects(data, budget, _phases="prioritize_all")
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )

print()
for budget in budgets():
    model_scores = algorithms.weighted_sampling(data, budget)
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )

print()
for budget in budgets():
    model_scores = algorithms.weighted_sampling(
        data,
        budget,
        epsilon=lambda rank, total: (1 if rank <= 2 else 0.5 if rank <= 5 else 0.1),
        # epsilon=lambda rank, total: 1/(rank + 1),
    )
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )

print()
model_scores_all = algorithms.statistical_ambiguity_reduction(
    data,
    budgets=budgets(),
    weight_ci_p=(0, 1),
)
for budget, model_scores in zip(budgets(), model_scores_all):
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(model_scores, model_scores_true):.3f} ",
        f"wTau: {utils.wtau(model_scores, model_scores_true):.3f} ",
        f"clup: {utils.clusters_p(model_scores):.3f}",
    )
