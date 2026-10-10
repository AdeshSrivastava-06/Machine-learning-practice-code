"""
Permutation test + Bootstrap CI + Cohen's d

A non-parametric way to compare two groups without assuming normality.
Uses train.csv (Titanic: Fare by Survived) if present, otherwise synthetic data.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- Load data ----------
try:
    df = pd.read_csv("train.csv")
    a = df.loc[df["Survived"] == 1, "Fare"].dropna().to_numpy()
    b = df.loc[df["Survived"] == 0, "Fare"].dropna().to_numpy()
    label_a, label_b, metric = "Survived", "Died", "Fare"
except Exception:
    a = rng.lognormal(3.2, 0.8, 200)
    b = rng.lognormal(2.9, 0.8, 250)
    label_a, label_b, metric = "Group A", "Group B", "Value"

obs_diff = a.mean() - b.mean()

# ---------- 1. Permutation test (two-sided) ----------
n_perm = 10_000
pooled = np.concatenate([a, b])
perm_diffs = np.empty(n_perm)
for i in range(n_perm):
    rng.shuffle(pooled)
    perm_diffs[i] = pooled[:len(a)].mean() - pooled[len(a):].mean()
p_value = (np.sum(np.abs(perm_diffs) >= abs(obs_diff)) + 1) / (n_perm + 1)

# ---------- 2. Bootstrap 95% CI for difference in means ----------
n_boot = 10_000
boot_diffs = np.array([
    rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean()
    for _ in range(n_boot)
])
ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

# ---------- 3. Cohen's d (effect size) ----------
pooled_sd = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                    / (len(a) + len(b) - 2))
cohens_d = obs_diff / pooled_sd
size = ("negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5
        else "medium" if abs(cohens_d) < 0.8 else "large")

# ---------- Results ----------
print(f"{metric}: {label_a} mean = {a.mean():.3f} | {label_b} mean = {b.mean():.3f}")
print(f"Observed difference : {obs_diff:.3f}")
print(f"Permutation p-value : {p_value:.4f}")
print(f"Bootstrap 95% CI    : [{ci_low:.3f}, {ci_high:.3f}]")
print(f"Cohen's d           : {cohens_d:.3f} ({size})")

# ---------- Plots 
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(perm_diffs, bins=50, color="steelblue", alpha=0.8)
ax[0].axvline(obs_diff, color="red", lw=2, label=f"Observed = {obs_diff:.2f}")
ax[0].set(title="Permutation null distribution", xlabel="Difference in means")
ax[0].legend()

ax[1].hist(boot_diffs, bins=50, color="seagreen", alpha=0.8)
ax[1].axvline(ci_low, color="black", ls="--")
ax[1].axvline(ci_high, color="black", ls="--", label="95% CI")
ax[1].set(title="Bootstrap distribution", xlabel="Difference in means")
ax[1].legend()

plt.tight_layout()
plt.show()
