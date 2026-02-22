# %%

import evaluation_bandit.utils_fig
import evaluation_bandit.utils
import matplotlib.pyplot as plt
import statistics


fig, (ax1, ax2) = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(4, 2.5),
)

cmap = plt.get_cmap("YlGn")

model_to_color = {
    # nice dark green
    "model A": "#080",
    "model B": "#880",
    # mix of green and red
    "model C": "#950",
    # dark red but not bright
    "model D": "#a00",
}


def plot_trace(ax, model_scores, plot_name=None):
    print(sum(len(v) for v in model_scores.values()))
    models = model_scores.keys()

    # sort models by final score
    models_sorted = sorted(
        models,
        key=lambda m: statistics.mean(model_scores[m]),
        reverse=True,
    )
    for model_i, model in enumerate(models_sorted):
        scores = [model_scores[model][: i + 1] for i in range(len(model_scores[model]))]
        color = model_to_color[model]
        ax.plot(
            [statistics.mean(v) for v in scores],
            label=model,
            color=color,
            linewidth=1,
            zorder=100 - model_i,
            marker="o",
        )
        ci = [
            evaluation_bandit.utils.confidence_interval(v, confidence=0.3)
            for v in scores
        ]
        ax.fill_between(
            range(len(scores)), [v[0] for v in ci], [v[1] for v in ci], color="#ccc"
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Evaluation item")
    if ax == ax1:
        ax.set_ylabel("Average score", labelpad=-3)
    ax.set_ylim(0, 100)
    ax.set_xlim(1.5, 9)
    ax.set_xticks(range(2, 9))
    if ax == ax1:
        ax.text(
            1,
            0.05,
            "Evaluation effort uniform\nbut spent on poor models",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
        )
        # text model names
        for model in models_sorted:
            ax.text(
                6.5,
                statistics.mean(model_scores[model]),
                model,
                fontsize=8,
                ha="left",
                va="center",
                color=model_to_color[model],
            )
    if ax == ax2:
        ax.text(
            1,
            1,
            "Budget focused on\ncomparing top models",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="top",
        )
        # add arrow pointing to model A
        ax.annotate(
            "",
            xy=(8.2, 70),
            xytext=(8.8, 90),
            arrowprops=dict(
                arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=-0.2"
            ),
        )
        ax.annotate(
            "",
            xy=(8.2, 80),
            xytext=(8.8, 90),
            arrowprops=dict(
                arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=-0.2"
            ),
        )

        ax.text(
            0.4,
            0.05,
            "Stopped early",
            transform=ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
        )
        ax.annotate(
            "",
            xy=(3.2, 38),
            xytext=(7, 12),
            arrowprops=dict(
                arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=0.2"
            ),
        )
        ax.annotate(
            "",
            xy=(5.2, 60),
            xytext=(7, 12),
            arrowprops=dict(
                arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=0.2"
            ),
        )


model_scores_all = {
    "model A": [92, 100, 50, 80, 80, 60, 100, 90, 90],
    "model B": [78, 100, 60, 100, 51, 70, 70, 40, 80],
    "model C": [45, 75, 40, 60, 80, 50, 50, 40],
    "model D": [10, 75, 40, 30, 50, 30, 70, 80],
}

model_scores_random = {k: v[:7] for k, v in model_scores_all.items()}

model_scores_bandit = {
    "model A": model_scores_all["model A"][:None],
    "model B": model_scores_all["model B"][:None],
    "model C": model_scores_all["model C"][:6],
    "model D": model_scores_all["model D"][:4],
}


plot_trace(ax1, model_scores_random, "random")
plot_trace(ax2, model_scores_bandit, "bandit")

plt.tight_layout(pad=0.1)
plt.savefig("../figures/intro_figure.svg")
plt.show()

# %%
import evaluation_bandit.utils_fig
import matplotlib.pyplot as plt

progress = {
    "model A": [1, 7, 10, 11, 13],
    "model B": [
        2,
        5,
        9,
        12,
    ],
    "model C": [3, 6],
    "model D": [4, 8],
}


def item_to_color(score):
    return plt.cm.RdYlGn(score / 100)


# have a grid 4x10 with some items (circles) filled left to right
plt.figure(figsize=(4, 2.5))

for model_i, (model, model_progress) in enumerate(progress.items()):
    for i, x in enumerate(model_progress):
        score = model_scores_all[model][i + 2]
        color = item_to_color(score)
        plt.plot(i, model_i, "o", markersize=18, color=color)
        plt.text(i, model_i, x, fontsize=10, ha="center", va="center")
    plt.scatter(
        i + 1,
        model_i,
        s=340,
        facecolors="lightgray",
        edgecolors="black",
        linewidths=1.5,
        linestyles=":",
    )


for model_i, i, offset_x, offset_y in [
    (0, 5, 0, 0.2),
    (1, 4, 0, 0.2),
    (2, 2, 0.3, 0),
    (3, 2, 0.3, 0),
]:
    plt.annotate(
        "",
        xy=(i + offset_x, model_i + offset_y),
        xytext=(4.4, 2.8),
        arrowprops=dict(
            arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=-0.1"
        ),
    )

# TODO: plot to the four last circles and describe it as "available model+item to evaluate next"
plt.annotate(
    "model A was\nevaluated last\n(step 13) on\nitem 5 with\nhigh score (green)",
    xy=(4, 0.2),  # slightly top right from the circle at (4, 0)
    xytext=(6, 2),
    arrowprops=dict(
        arrowstyle="->", color="black", lw=1, connectionstyle="arc3,rad=-0.1"
    ),
    fontsize=8,
    ha="left",
    va="bottom",
)

plt.text(
    4.5,
    3,
    "available model+item\nfor $\\pi$ to evaluate next",
    fontsize=8,
    ha="left",
    va="center",
)


# TODO: point to last model A's item (13) and say "the last evaluated item (13th) is model A on item 5 with high score (green)"


plt.xlim(-0.5, 9.2)
plt.ylim(-0.3, len(progress) - 0.7)
plt.xticks(range(10), range(1, 11))
plt.yticks(range(len(progress)), progress.keys())
plt.gca().invert_yaxis()
plt.xlabel("Evaluation item")
plt.gca().spines[["top", "right"]].set_visible(False)
plt.tight_layout(pad=0.1)
plt.savefig("../figures/intro_figure_constrained.svg")
plt.show()
