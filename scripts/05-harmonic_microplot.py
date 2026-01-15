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
plt.savefig("../figures/harmonic_microplot_pow1.svg", transparent=True)
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
plt.savefig("../figures/harmonic_microplot_pow3.svg", transparent=True)
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
plt.savefig("../figures/harmonic_microplot_top.svg", transparent=True)
plt.show()
