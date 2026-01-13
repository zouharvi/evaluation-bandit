# %%

from translation_tournament import algorithms, utils
import importlib
importlib.reload(algorithms)
importlib.reload(utils)
import numpy as np

# ['cs-de_DE', 'cs-uk_UA', 'en-ar_EG', 'en-bho_IN', 'en-cs_CZ', 'en-et_EE', 'en-is_IS', 'en-it_IT', 'en-ja_JP', 'en-ko_KR', 'en-mas_KE', 'en-ru_RU', 'en-sr_Cyrl_RS', 'en-uk_UA', 'en-zh_CN', 'ja-zh_CN'])

data = utils.load_data_single()
for item in data:
    for i in range(3):
        item["scores"] |= {
            model+"*"*i: score + np.random.normal(0, 0.0001)
            for model, score in item["scores"].items()
        }

ranking_true = utils.items_to_model_scores(data, average=True)
budgets = lambda: np.linspace(200, int(len(data)*len(data[0]["scores"])*0.5), 5, dtype=int)

for budget in budgets():
    ranking = algorithms.baseline(data, budget)
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(ranking, ranking_true):.3f} ",
        f"wTau: {utils.wtau(ranking, ranking_true):.3f}",
    )

print()
for budget in budgets():
    ranking = algorithms.successive_rejects(data, budget, phases="constant")
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(ranking, ranking_true):.3f} ",
        f"wTau: {utils.wtau(ranking, ranking_true):.3f}",
    )

print()
for budget in budgets():
    ranking = algorithms.epsilon_greedy(data, budget, topk=3, epsilon=0.5)
    print(
        f"Budget: {budget:>4}",
        f"Tau:  {utils.tau(ranking, ranking_true):.3f} ",
        f"wTau: {utils.wtau(ranking, ranking_true):.3f}",
    )