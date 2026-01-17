# %%
import matplotlib.pyplot as plt
import numpy as np

# Data generation
x_dots = np.arange(1, 15)
y_dots = 1 / x_dots

# Plotting
fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

# Little dots on natural numbers
ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

# No decorations, Tufte style (remove spines and ticks)
ax.axis("off")

# Save as SVG
plt.tight_layout(pad=0)
plt.savefig("../figures/microplot_weight_pow1.svg", transparent=True)
plt.show()

# Data generation
x_dots = np.arange(1, 15)
y_dots = 1 / x_dots**2

# Plotting
fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

# Little dots on natural numbers
ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

# No decorations, Tufte style (remove spines and ticks)
ax.axis("off")

# Save as SVG
plt.tight_layout(pad=0)
plt.savefig("../figures/microplot_weight_pow3.svg", transparent=True)
plt.show()


# Data generation
x_dots = np.arange(1, 15)
y_dots = [1 if i <= 3 else 0 for i in x_dots]

# Plotting
fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

# Little dots on natural numbers
ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

# No decorations, Tufte style (remove spines and ticks)
ax.axis("off")

# Save as SVG
plt.tight_layout(pad=0)
plt.savefig("../figures/microplot_weight_top.svg", transparent=True)
plt.show()

# %%

import translation_bandit.utils_fig
import translation_bandit.utils
import matplotlib.pyplot as plt
import statistics

data_all = translation_bandit.utils.load_data()
data_all = [
    [
        statistics.mean([item["scores"][model] for item in data])
        for model in data[0]["scores"].keys()
    ]
    for data in data_all.values()
]

data_all = [
    [100 + 4 * x for x in data] if statistics.mean(data) < 0 else data
    for data in data_all
]
data_all = [
    x for data in data_all for x in data
]

fig, ax = plt.subplots(figsize=(1, 0.3))
plt.hist(data_all, color="black")
ax.axis("off")
plt.tight_layout(pad=0)
plt.savefig("../figures/microplot_model_avg.svg", transparent=True)
plt.show()
