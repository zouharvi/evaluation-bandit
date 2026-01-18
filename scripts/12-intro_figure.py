# %%

import translation_bandit.utils_fig
import translation_bandit.utils
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
    "model C": "#a40",
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
            translation_bandit.utils.confidence_interval(v, confidence=0.3)
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
    "model A": [90, 100, 50, 80, 80, 60, 100, 90, 90],
    "model B": [80, 90, 60, 100, 50, 70, 70, 40, 80],
    "model C": [45, 80, 40, 60, 80, 50, 50, 40],
    "model D": [10, 70, 40, 30, 50, 30, 70, 80],
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
