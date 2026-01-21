# %%

import evaluation_bandit.utils
import random
import statistics

# ict_keys(['wmt25_cs-de_DE', 'wmt25_cs-uk_UA', 'wmt25_en-ar_EG', 'wmt25_en-bho_IN', 'wmt25_en-cs_CZ', 'wmt25_en-et_EE', 'wmt25_en-is_IS', 'wmt25_en-it_IT', 'wmt25_en-ja_JP', 'wmt25_en-ko_KR', 'wmt25_en-mas_KE', 'wmt25_en-ru_RU', 'wmt25_en-sr_Cyrl_RS', 'wmt25_en-uk_UA', 'wmt25_en-zh_CN', 'wmt25_ja-zh_CN', 'wmt24_cs-uk', 'wmt24_en-cs', 'wmt24_en-de', 'wmt24_en-es', 'wmt24_en-hi', 'wmt24_en-is', 'wmt24_en-ja', 'wmt24_en-ru', 'wmt24_en-uk', 'wmt24_en-zh', 'wmt24_ja-zh', 'wmt23.sent_en-de', 'wmt23_cs-uk', 'wmt23_de-en', 'wmt23_en-cs', 'wmt23_en-de', 'wmt23_en-ja', 'wmt23_en-zh', 'wmt23_he-en', 'wmt23_ja-en', 'wmt23_zh-en'])
data = evaluation_bandit.utils.load_data()["wmt25_cs-de_DE"]
random.Random(0).shuffle(data)


models = list(data[0]["scores"].keys())
model_scores = {model: [(0, data[0]["scores"][model])] for model in models}
model_counts = {model: 1 for model in models}

model_ranks = {}
models_sorted = sorted(models, key=lambda m: model_scores[m][-1][1], reverse=True)
for rank, model in enumerate(models_sorted):
    model_ranks[model] = [rank]

step = 0
while True:
    if step >= 2000:
        break
    for model in models:
        if model_counts[model] >= len(data):
            break
        entry = data[model_counts[model]]
        model_counts[model] += 1
        new_avg = statistics.mean(
            [score for s, score in model_scores[model]] + [entry["scores"][model]]
        )
        model_scores[model].append((step, new_avg))

        models_sorted = sorted(
            models, key=lambda m: model_scores[m][-1][1], reverse=True
        )
        for rank, model in enumerate(models_sorted):
            model_ranks[model].append(rank)
        step += 1
    else:
        continue
    break

model_scores_random = model_scores
model_ranks_random = model_ranks
model_counts_random = model_counts


# stochastic sampling based on ranksmooth
models = list(data[0]["scores"].keys())

# Initialize with 1 sample
raw_scores = {model: [data[0]["scores"][model]] for model in models}
model_means = {model: raw_scores[model][0] for model in models}

model_scores = {model: [(0, model_means[model])] for model in models}


sorted_models_init = sorted(models, key=lambda m: model_means[m], reverse=True)
model_ranks = {model: [rank] for rank, model in enumerate(sorted_models_init)}

model_counts = {model: 1 for model in models}
step = 0
total_items = len(data)

r_object = random.Random(1)

# cold start
for _ in range(10):
    for model in models:
        entry = data[model_counts[model]]
        model_counts[model] += 1
        new_avg = statistics.mean(
            [score for s, score in model_scores[model]] + [entry["scores"][model]]
        )
        model_scores[model].append((step, new_avg))

        models_sorted = sorted(
            models, key=lambda m: model_scores[m][-1][1], reverse=True
        )
        for rank, model in enumerate(models_sorted):
            model_ranks[model].append(rank)
        step += 1

while True:
    if step >= 2000:
        break

    # Active models are those who haven't processed all data
    active_models = [m for m in models if model_counts[m] < total_items]
    if not active_models:
        break

    # Sort active models by mean score to determine rank for sampling
    active_models.sort(key=lambda m: model_means[m], reverse=True)

    # Sample a model
    chosen = r_object.choices(
        active_models,
        weights=[1.0 / (rank + 1) for rank in range(len(active_models))],
        k=1,
    )[0]

    # Update the chosen model locally
    idx = model_counts[chosen]
    score = data[idx]["scores"][chosen]
    raw_scores[chosen].append(score)
    model_counts[chosen] += 1

    # Update mean
    new_mean = statistics.mean(raw_scores[chosen])
    model_means[chosen] = new_mean

    # Record trace
    step += 1
    model_scores[chosen].append((step, new_mean))

    # Update ranks for ALL models (for global ranking visualization)
    models_sorted_global = sorted(models, key=lambda m: model_means[m], reverse=True)
    for rank, m in enumerate(models_sorted_global):
        model_ranks[m].append(rank)


model_scores_sampling = model_scores
model_ranks_sampling = model_ranks
model_counts_sampling = model_counts

# %%

import evaluation_bandit.utils_fig
import matplotlib.pyplot as plt


def plot_trace(model_scores, model_ranks, model_counts, plot_name=None):
    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(9, 2.5),
        gridspec_kw={"width_ratios": [3, 3, 1]},
    )

    models = model_scores.keys()

    # get colormap "hot"
    cmap = plt.get_cmap("RdYlGn")

    # sort models by final score
    models_sorted = sorted(
        models,
        key=lambda m: statistics.mean([item["scores"][m] for item in data]),
        reverse=True,
    )
    for model_i, model in enumerate(models_sorted):
        steps, scores = zip(*model_scores[model])
        ranks = model_ranks[model]
        # based on model_i/len(models_sorted) get from cmap
        color = cmap(1 - model_i / len(models_sorted))
        ax1.plot(
            steps,
            scores,
            label=model,
            color=color,
            linewidth=1,
            zorder=100 - model_i,
        )
        ax2.plot(
            [r + 1 for r in ranks],
            label=model,
            color=color,
            linewidth=1,
            zorder=100 - model_i,
        )
    model_i_local = sorted(models, key=lambda m: model_scores[m][-1][1], reverse=True)
    model_i_local = {m: i for i, m in enumerate(model_i_local)}
    for model_i, model in enumerate(models_sorted):
        color = cmap(1 - model_i_local[model] / len(models_sorted))
        ax3.barh(
            model_i,
            model_counts[model],
            color=color,
            zorder=100 - model_i,
        )
        ax3.text(
            5,
            model_i - 0.05,
            model,
            va="center",
            fontsize=6,
            color="black"
            if model_i_local[model] > 3 and model_i_local[model] < 18
            else "white",
            zorder=100 - model_i,
        )

    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.set_xlim(0, 250)
    ax3.set_xticks([0, 100, 200])

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Evaluation step")
        ax.set_xlim(-5, 0.4 * len(data) * len(models))

    ax1.set_ylabel("Average score", labelpad=-5)
    ax2.set_ylabel("Ranking", labelpad=-1)
    ax3.set_xlabel("Evaluation focus   ")
    ax1.set_ylim(60, 100)
    ax2.invert_yaxis()
    ax3.invert_yaxis()
    ax2.set_yticks([1, 5, 10, 15, 20])
    ax2.set_yticklabels([1, 5, 10, 15, 20])
    ax3.set_yticks([])

    plt.tight_layout(pad=0.4)
    if plot_name is not None:
        plt.savefig(f"../figures/trace_plot_{plot_name}.svg", transparent=True)
    plt.show()


plot_trace(model_scores_random, model_ranks_random, model_counts_random, "random")
plot_trace(
    model_scores_sampling, model_ranks_sampling, model_counts_sampling, "sampling"
)
