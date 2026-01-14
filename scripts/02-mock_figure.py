# %%

data = {
    "A": [
        55, 65, 75, 85,  10, 95, 70, 60,  50, 50, 70, 80, 100, 100, 80, 80, 60, 70, 95, 90,
    ],
    "B": [
        62, 68, 79, 88,  15, 97, 73, 63,  57, 42, 77, 83, 10, 93, 87, 82, 70, 65, 90, 85,
    ],
    "C": [
        62, 45, 78, 90,  55, 43, 23, 67,  89, 62, 34, 56, 20, 90, 70, -90,
    ],
    "D": [
        70, 81, 40, 20,  50, 60, 60, 60, 50, 40, 50, -60,
    ],
    "E": [
        40, 20, 70, 50,  63, 44, 60, -40,
    ],
    "F": [
        30, 50, 30, -30,
    ],
}

COLORS = [
    "#020",
    "#242",
    "#464",
    "#686",
    "#8A8",
    "#ACA",
]
MODEL_COLOR = {model: COLORS[i] for i, model in enumerate(data.keys())}

import matplotlib.pyplot as plt
import translation_bandit.utils_fig
import statistics
import numpy as np

data_new = {}
for model, values in data.items():
    values_new = []
    for x_i, x in enumerate(values):
        values_new.append(abs(x) + np.random.normal(0, 3) + np.random.normal(0, 1) * (x_i / len(values)) * 10)
        values_new.append(x + np.random.normal(0, 3) + np.random.normal(0, 1) * (x_i / len(values)) * 10)
        
    data_new[model] = values_new

data = data_new


plt.figure(figsize=(4, 2.5))

for model, model_values in data.items():
    plt.plot(
        [
            statistics.mean(model_values[: i + 1])
            for i, v in enumerate(model_values)
            if v >= 0
        ],
        marker=".",
        color=MODEL_COLOR[model],
        markersize=6,
    )
    if any(x < 0 for x in model_values):

        plt.plot(
            [len(model_values)-2, len(model_values) - 1],
            [statistics.mean([abs(x) for x in model_values[:-1]]),
             statistics.mean([abs(x) for x in model_values])
            ],
            marker=None,
            color=MODEL_COLOR[model],
        )

        plt.scatter(
            [len(model_values) - 1],
            [statistics.mean([abs(x) for x in model_values])],
            marker="x",
            color="#D44",
            s=70,
            zorder=100,
        )

plt.ylabel("Models", labelpad=-4)
plt.xlabel("Annotation item", labelpad=1)
plt.ylim(0, 100)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0)
plt.savefig("/home/vilda/Downloads/mock_successive_rejects.svg")
plt.show()

# %%

import random
import matplotlib.pyplot as plt
import translation_bandit.utils_fig
import numpy as np
import statistics

plt.figure(figsize=(4, 2))

MODELS = {
    "A": 75,
    "B": 72,
    "C": 65,
    "D": 60,
    "E": 55,
    "F": 40,
}

ITEMS = np.random.normal(0, 15, 400).tolist()
EPSILON = 0.4

data = {
    model: [
        np.random.normal(MODELS[model], 20)
        for _ in range(5)
    ] for model in MODELS.keys()
}

# TODO: 5 points at the beginning
for model in MODELS.keys():
    plt.plot(
        list(range(5)),
        [statistics.mean(data[model][:i+1]) for i in range(5)],
        marker=".",
        color=MODEL_COLOR[model],
        markersize=7,
    )


# simulare epsilon-greedy
for i in range(5, 200):
    if random.random() > EPSILON:
        # sort models by performance
        models = sorted(list(MODELS.keys()), key=lambda m: np.mean(data[m]), reverse=True)[:3]
    else:
        models = list(MODELS.keys())

    model = random.choice(models)
    item = ITEMS[len(data[model])]
    performance = MODELS[model] + item + np.random.normal(0, 20)
    data[model].append(performance)

    # value continues from previous one
    for _model in MODELS.keys():
        if model == _model:
            continue
        last_value = statistics.mean(data[_model])
        plt.plot(
            [i-1, i],
            [last_value, last_value],
            color=MODEL_COLOR[_model],
        )
    plt.plot(
        [i-1, i],
        [
            statistics.mean(data[model][:-1]),
            statistics.mean(data[model]),
        ],
        color=MODEL_COLOR[model],
    )
    plt.scatter(
        [i],
        [statistics.mean(data[model])],
        marker=".",
        color=MODEL_COLOR[model],
        s=50,
    )


plt.ylabel("Models", labelpad=-4)
plt.xlabel("Annotation progression", labelpad=1)
plt.ylim(0, 100)
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0)
plt.savefig("/home/vilda/Downloads/mock_epsilon_greedy.svg")
plt.show()