from . import utils
import numpy as np

def simulate(
    fn,
    accepts_budgets=False,
):
    output = []
    for data_name, data in utils.load_data().items():
        if data_name != "en-ko_KR":
            continue
        budgets = budgets = lambda: np.linspace(
            200, int(len(data) * len(data[0]["scores"]) * 0.5), 10, dtype=int
        )
        if accepts_budgets:
            model_scores_all = fn(data, budgets=budgets())
        else:
            model_scores_all = [
                fn(data, budget) for budget in budgets()
            ]
        model_scores_true = {
           model: [item["scores"][model] for item in data] for model in data[0]["scores"]
        }
    
        for budget, model_scores in zip(budgets(), model_scores_all):
            output.append({
                "data_name": data_name,
                "budget": budget,
                "tau": utils.tau(model_scores, model_scores_true),
                "wtau": utils.wtau(model_scores, model_scores_true),
                "clup": utils.clusters_p(model_scores),
            })
    return output