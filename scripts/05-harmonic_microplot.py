# %%
import matplotlib.pyplot as plt
import numpy as np

# Data generation
x_line = np.linspace(1, 8, 500)
y_line = 1 / x_line

x_dots = np.arange(1, 8 + 1)
y_dots = 1 / x_dots

# Plotting
fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

# Black line
ax.plot(x_line, y_line, color="black", linewidth=1.5)

# Little dots on natural numbers
ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

# No decorations, Tufte style (remove spines and ticks)
ax.axis("off")

# Save as SVG
plt.tight_layout(pad=0)
plt.savefig("../figures/harmonic_microplot.svg", transparent=True)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np

# Data generation
x_line = np.linspace(1, 8, 500)
y_line = 1 / x_line**0.3

x_dots = np.arange(1, 8 + 1)
y_dots = 1 / x_dots**0.3

# Plotting
fig, ax = plt.subplots(figsize=(1, 0.3))  # Small size for microplot

# Black line
ax.plot(x_line, y_line, color="black", linewidth=1.5)

# Little dots on natural numbers
ax.scatter(x_dots, y_dots, color="black", s=20, zorder=3, marker=".")

# No decorations, Tufte style (remove spines and ticks)
ax.axis("off")

# Save as SVG
plt.tight_layout(pad=0)
plt.savefig("../figures/harmonic_microplot_pow3.svg", transparent=True)
plt.show()
