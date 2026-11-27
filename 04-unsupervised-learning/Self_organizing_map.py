# Self-Organizing Map (SOM) on the Wine dataset.
# Goal: place 178 wines with 13 chemical features on a small 2D grid of neurons,
# so that similar wines land on the same neuron or on neighbouring neurons.
# The true wine class is NOT used for training, only to check the result.
#
# How a SOM learns:
#   1. Pick a sample and find the neuron whose weights are closest to it (the BMU)
#   2. Pull the BMU and its grid neighbours towards the sample
#   3. Slowly reduce the learning rate and the neighbourhood size
#
# Part 1 builds the SOM from scratch with numpy.
# Part 2 uses the minisom library (pip install minisom).
# Part 3 compares the two and plots the U-matrix, which shows the distance
# between neighbouring neurons. Bright areas are neurons that are similar to
# their neighbours (the inside of a cluster) and dark areas are the borders
# between clusters.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

GRID = 8
ITERATIONS = 5000
SEED = 42

X, y = load_wine(return_X_y=True)
X = StandardScaler().fit_transform(X)
n_samples, n_features = X.shape
print("Dataset:", n_samples, "wines,", n_features, "features, 3 true classes")
print(f"Grid: {GRID} x {GRID} = {GRID * GRID} neurons, iterations: {ITERATIONS}")
print()

# ---------- Part 1: SOM from scratch ----------
class SimpleSOM:
    def __init__(self, rows, cols, n_features, seed=0):
        self.rows, self.cols = rows, cols
        self.rng = np.random.RandomState(seed)
        self.weights = None
        self.n_features = n_features
        # Position of every neuron on the grid, used to measure grid distance
        self.grid = np.array([[r, c] for r in range(rows) for c in range(cols)])

    def init_weights(self, data):
        # Start from randomly chosen samples
        idx = self.rng.choice(len(data), self.rows * self.cols, replace=False)
        self.weights = data[idx].copy()

    def winner(self, x):
        return int(np.argmin(((self.weights - x) ** 2).sum(axis=1)))

    def train(self, data, iterations, lr_start=0.5, lr_end=0.01, sigma_end=0.5):
        sigma_start = max(self.rows, self.cols) / 2
        for t in range(iterations):
            fraction = t / iterations
            # Learning rate and neighbourhood size shrink step by step
            lr = lr_start * (lr_end / lr_start) ** fraction
            sigma = sigma_start * (sigma_end / sigma_start) ** fraction
            x = data[self.rng.randint(len(data))]
            bmu = self.winner(x)
            grid_distance = ((self.grid - self.grid[bmu]) ** 2).sum(axis=1)
            influence = np.exp(-grid_distance / (2 * sigma ** 2))
            self.weights += lr * influence[:, None] * (x - self.weights)

    def quantization_error(self, data):
        # Average distance between a sample and its winning neuron
        return np.mean([np.linalg.norm(x - self.weights[self.winner(x)]) for x in data])

    def topographic_error(self, data):
        # Share of samples whose two closest neurons are not neighbours on the grid
        errors = 0
        for x in data:
            first, second = np.argsort(((self.weights - x) ** 2).sum(axis=1))[:2]
            if np.linalg.norm(self.grid[first] - self.grid[second]) >= 1.5:
                errors += 1
        return errors / len(data)

    def u_matrix(self):
        w = self.weights.reshape(self.rows, self.cols, -1)
        result = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                distances = [np.linalg.norm(w[r, c] - w[r + dr, c + dc])
                             for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                             if (dr or dc) and 0 <= r + dr < self.rows and 0 <= c + dc < self.cols]
                result[r, c] = np.mean(distances)
        return result

def class_map_purity(winner_positions, labels):
    # Each neuron gets the majority class of the wines that landed on it.
    # Purity is the share of wines whose class matches their neuron's class.
    correct, used = 0, set()
    for pos in set(winner_positions):
        members = labels[[i for i, p in enumerate(winner_positions) if p == pos]]
        correct += np.bincount(members).max()
        used.add(pos)
    return correct / len(labels), len(used)

som = SimpleSOM(GRID, GRID, n_features, seed=SEED)
som.init_weights(X)
qe_before, te_before = som.quantization_error(X), som.topographic_error(X)
som.train(X, ITERATIONS)
qe_scratch, te_scratch = som.quantization_error(X), som.topographic_error(X)
scratch_positions = [tuple(som.grid[som.winner(x)]) for x in X]
purity_scratch, used_scratch = class_map_purity(scratch_positions, y)

print("Part 1: SOM from scratch")
print(f"{'':28s} {'Before':>8s} {'After':>8s}")
print(f"{'Quantization error':28s} {qe_before:8.3f} {qe_scratch:8.3f}")
print(f"{'Topographic error':28s} {te_before:8.3f} {te_scratch:8.3f}")
print()

# ---------- Part 2: SOM with minisom ----------
print("Part 2: SOM with the minisom library")
try:
    from minisom import MiniSom
except ImportError:
    print("minisom is not installed. Run: pip install minisom")
    MiniSom = None

if MiniSom is not None:
    lib = MiniSom(GRID, GRID, n_features, sigma=GRID / 2, learning_rate=0.5, random_seed=SEED)
    lib.random_weights_init(X)
    lib.train_random(X, ITERATIONS)
    qe_lib, te_lib = lib.quantization_error(X), lib.topographic_error(X)
    lib_positions = [lib.winner(x) for x in X]
    purity_lib, used_lib = class_map_purity(lib_positions, y)
    print(f"Quantization error: {qe_lib:.3f}, topographic error: {te_lib:.3f}")
    print()

    # ---------- Part 3: comparison ----------
    print("Part 3: comparison")
    print(f"{'':28s} {'Scratch':>8s} {'minisom':>8s}")
    print(f"{'Quantization error':28s} {qe_scratch:8.3f} {qe_lib:8.3f}")
    print(f"{'Topographic error':28s} {te_scratch:8.3f} {te_lib:8.3f}")
    print(f"{'Neurons that won a wine':28s} {used_scratch:8d} {used_lib:8d}")
    print(f"{'Class purity':28s} {purity_scratch:8.3f} {purity_lib:8.3f}")

# ---------- Plot the U-matrix with the wines on top ----------
def draw(ax, u_matrix, positions, title):
    ax.imshow(u_matrix, cmap="bone_r", origin="lower")
    colors = ["tab:red", "tab:green", "tab:blue"]
    jitter = np.random.RandomState(0).uniform(-0.3, 0.3, (len(positions), 2))
    for (r, c), label, j in zip(positions, y, jitter):
        ax.plot(c + j[1], r + j[0], "o", color=colors[label], markersize=4)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

panels = 2 if MiniSom is not None else 1
fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 5.5))
axes = np.atleast_1d(axes)
draw(axes[0], som.u_matrix(), [(p[0], p[1]) for p in scratch_positions], "Scratch SOM: U-matrix")
if MiniSom is not None:
    draw(axes[1], lib.distance_map(), lib_positions, "minisom: U-matrix")
fig.suptitle("Wine classes on the SOM (red, green, blue are the three classes)")
plt.tight_layout()
plt.show()

# Summary
print()
print("Summary:")
print(f"The topographic error fell from {te_before:.2f} to {te_scratch:.2f}, so the map is now ordered:")
print("neighbouring neurons hold similar wines.")
if qe_scratch > qe_before:
    print(f"The quantization error rose from {qe_before:.2f} to {qe_scratch:.2f}. This is expected here,")
    print("because the neurons started as real wines. Training pulls neighbours together,")
    print("which smooths the map and costs a little accuracy per wine.")
else:
    print(f"The quantization error fell from {qe_before:.2f} to {qe_scratch:.2f}.")
print(f"The wines landed on {used_scratch} of {GRID * GRID} neurons, and the class purity is "
      f"{purity_scratch:.2f}, even though the classes were never used for training.")
if purity_scratch > 0.85:
    print("Wines of the same class sit close together on the map, so the SOM found the structure.")
print("Purity is measured on the same data used for training, so it is an optimistic number.")
print("A SOM keeps the neighbourhood: nearby neurons hold similar wines, which is the main")
print("difference from KMeans, where the cluster numbers have no order.")
