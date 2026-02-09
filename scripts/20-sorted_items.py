# %%

from evaluation_bandit import utils
import statistics

data = utils.load_data(wmt_years={"wmt25"})["wmt25_cs-de_DE"]
data.sort(key=lambda x: statistics.mean(x["scores"].values()))

# %%

import matplotlib.pyplot as plt
from evaluation_bandit import utils_fig

models = list(data[0]["scores"].keys())
models = ["Gemini-2.5-Pro", "Gemma-3-27B", "Yolu"]

WINDOW_SIZE = 8


def average_smooth(ys):
    return [
        statistics.mean(ys[i : i + WINDOW_SIZE])
        for i in range(len(ys) - WINDOW_SIZE + 1)
    ]


plt.figure(figsize=(3.5, 2))
for model_i, model in enumerate(models):
    plt.plot(
        range(len(data) + 1 - WINDOW_SIZE),
        average_smooth([x["scores"][model] for x in data]),
        label=f"Model $m_{model_i + 1}$",
        alpha=0.5,
    )

plt.plot(
    range(len(data)),
    [statistics.mean(x["scores"].values()) for x in data],
    label="Average",
    color="black",
    linewidth=2,
)

plt.ylabel(r"Score $r_x$", labelpad=-4)
plt.xlabel(r"Evaluation item $x_i$")
plt.gca().spines[["right", "top"]].set_visible(False)
plt.legend(
    frameon=False,
    handlelength=0.9,
    handletextpad=0.2,
    labelspacing=0.2,
)
plt.show()

# %%

# rank models
print(
    sorted(
        data[0]["scores"].keys(),
        key=lambda m: statistics.mean(x["scores"][m] for x in data),
    )
)

[
    "Laniqo",
    "IRB-MT",
    "Yolu",
    "SRPOL",
    "Gemma-3-12B",
    "CUNI-MH-v2",
    "UvA-MT",
    "TowerPlus-9B",
    "Algharb",
    "Wenyiil",
    "Gemma-3-27B",
    "GemTrans",
    "CommandA-MT",
    "CommandA",
    "Mistral-Medium",
    "Shy",
    "DeepSeek-V3",
    "Claude-4",
    "GPT-4.1",
    "Gemini-2.5-Pro",
]
