"""
Pearson vs Spearman Correlation

Both measure the relationship between two numeric variables, but
differently:

Pearson Correlation:
    - Measures LINEAR relationship only.
    - Sensitive to outliers (a few extreme points can distort it a lot).
    - Assumes roughly normal, continuous data.
    - Range: -1 (perfect negative linear) to +1 (perfect positive linear).

Spearman Correlation:
    - Measures MONOTONIC relationship (as X increases, does Y
      consistently increase/decrease, even if not in a straight line?).
    - Works on RANKS of the data, not raw values -> more robust to
      outliers and works for non-linear (but monotonic) relationships.
    - Also used for ordinal data (rankings, ratings).

Rule of thumb: if relationship looks curved but still one-directional,
or data has outliers, Spearman is often more reliable than Pearson.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr

rng = np.random.RandomState(42)
n = 150


# Case 1: Clean linear relationship
# Both Pearson and Spearman should agree closely here.

x1 = rng.uniform(0, 10, n)
y1 = 2 * x1 + rng.normal(0, 1.5, n)

pearson_r1, pearson_p1 = pearsonr(x1, y1)
spearman_r1, spearman_p1 = spearmanr(x1, y1)

print("Case 1: Linear relationship")
print(f"  Pearson:  r={pearson_r1:.4f}, p={pearson_p1:.6f}")
print(f"  Spearman: r={spearman_r1:.4f}, p={spearman_p1:.6f}")


# Case 2: Non-linear but monotonic relationship (exponential)
# Pearson will underestimate the strength since it's curved;
# Spearman should capture it better since it only cares about
# "does Y consistently increase as X increases", not the exact shape.

x2 = rng.uniform(0, 5, n)
y2 = np.exp(x2) + rng.normal(0, 5, n)

pearson_r2, pearson_p2 = pearsonr(x2, y2)
spearman_r2, spearman_p2 = spearmanr(x2, y2)

print("\nCase 2: Non-linear (exponential) but monotonic relationship")
print(f"  Pearson:  r={pearson_r2:.4f}, p={pearson_p2:.6f}")
print(f"  Spearman: r={spearman_r2:.4f}, p={spearman_p2:.6f}")
print("  (Spearman should be noticeably higher since it captures the")
print("   monotonic trend regardless of the curve's exact shape)")


# Case 3: Linear relationship WITH an outlier
# Pearson gets dragged around by the outlier; Spearman (rank-based)
# is much more robust since the outlier just becomes "the highest rank".

x3 = rng.uniform(0, 10, n)
y3 = 2 * x3 + rng.normal(0, 1.5, n)
# inject one extreme outlier
x3 = np.append(x3, 9.5)
y3 = np.append(y3, 150)  # way off the trend

pearson_r3, pearson_p3 = pearsonr(x3, y3)
spearman_r3, spearman_p3 = spearmanr(x3, y3)

print("\nCase 3: Linear relationship + one extreme outlier")
print(f"  Pearson:  r={pearson_r3:.4f}, p={pearson_p3:.6f}")
print(f"  Spearman: r={spearman_r3:.4f}, p={spearman_p3:.6f}")
print("  (Pearson should drop noticeably due to the outlier;")
print("   Spearman should stay closer to the original Case 1 value)")


# Visualize all three cases

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

datasets = [
    (x1, y1, pearson_r1, spearman_r1, "Linear relationship"),
    (x2, y2, pearson_r2, spearman_r2, "Non-linear (exponential)\nbut monotonic"),
    (x3, y3, pearson_r3, spearman_r3, "Linear + outlier"),
]

for ax, (x, y, pr, sr, title) in zip(axes, datasets):
    ax.scatter(x, y, alpha=0.6, s=25, color="steelblue")
    ax.set_title(f"{title}\nPearson r={pr:.3f} | Spearman r={sr:.3f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pearson_vs_spearman.png", dpi=120)
plt.show()


# Bonus: Correlation matrix / heatmap for multiple variables

df = pd.DataFrame({
    "linear_x": x1,
    "linear_y": y1,
    "exp_x": x2,
    "exp_y": y2,
})

pearson_matrix = df.corr(method="pearson")
spearman_matrix = df.corr(method="spearman")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, matrix, name in zip(axes, [pearson_matrix, spearman_matrix], ["Pearson", "Spearman"]):
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.columns)
    ax.set_title(f"{name} Correlation Matrix")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("correlation_matrix_comparison.png", dpi=120)
plt.show()
