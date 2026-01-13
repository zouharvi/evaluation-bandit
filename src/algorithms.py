import math
import random
from typing import Literal
from . import utils

def baseline(data, budget):
    items = random.sample(
        data,
        k=budget//len(data[0]["scores"])
    )
    # pprint.pprint(
    #     sorted(utils.items_to_model_scores(items, average=True).items(), key=lambda x: x[1], reverse=True
    # ))

    return utils.items_to_model_scores(items, average=True)


def successive_rejects(data, budget, phases: Literal["constant", "flexible"] = "constant"):
    # shallow copy
    data = list(data)
    models = list(data[0]["scores"])
    if phases == "constant":
        phase_items = 2 * budget / (len(models) ** 2 + len(models) -2)
    elif phases == "flexible":
        raise ValueError("Not implemented yet.")
        # this is wrong
        phase_items = budget / (len(models) - 2)
    else:
        raise ValueError(f"Other (e.g. more dynamic) phase lengths not implemented yet.")
    
    # last phase needs to have at least two models still

    cost = 0

    model_phase_elimintation = {}
    model_scores = {model: [] for model in models}
    for phase in range(len(models) - 1):
        for _ in range(max(1, math.ceil(phase_items))):
            if not data:
                break
            item = data.pop(0)
            for model in models:
                cost += 1
                model_scores[model].append(item["scores"][model])
        # eliminate worst model
        model = min(
            models,
            key=lambda m: utils.statistics.mean(model_scores[m]),
        )
        models.remove(model)
        model_phase_elimintation[model] = phase
    # add last model
    model_phase_elimintation[models[0]] = phase + 1

    # return model_phase_elimintation
    # pprint.pprint(model_phase_elimintation)

    # print(cost, f"{phase_items:.1f}", len(data[0]["scores"]))
    # or model_phase_elimination?
    return {
        model: utils.statistics.mean(model_scores[model])
        for model in model_scores
    }


def epsilon_greedy(data, budget, topk=3, epsilon=0.5):
    model_index = {model: 0 for  model in data[0]["scores"]}
    model_scores = {model: [] for model in data[0]["scores"]}
    models = list(data[0]["scores"])
    cost = 0
    # initial phase
    for _ in range(3):
        for model in models:
            item = data[model_index[model]]
            model_scores[model].append(item["scores"][model])
            model_index[model] += 1
            cost += 1

    while cost < budget:
        if random.random() < epsilon:
            # explore: select model with least evaluations
            model = min(models, key=model_index.get)
        else:
            # exploit top-k
            model = random.choice(models[:topk])
        item = data[model_index[model]]
        model_scores[model].append(item["scores"][model])
        model_index[model] += 1
        cost += 1

        models = [
            model
            for model, i in model_index.items()
            if i < len(data)
        ]
        models.sort(key=lambda m: utils.statistics.mean(model_scores[m]), reverse=True)

    return {
        model: utils.statistics.mean(model_scores[model])
        for model in model_scores
    }