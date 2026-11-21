# Color quantization of a real image with KMeans and MiniBatchKMeans.
# Goal: reduce the number of colors in a photo by clustering its pixels.
# Every pixel is a point with three values (red, green, blue). The cluster
# centres become the color palette, and each pixel is replaced by the colour
# of its cluster centre. This is a practical use of clustering for image
# compression.
# Three methods are compared for each palette size:
#   1. KMeans fitted on all pixels
#   2. KMeans fitted on a random sample of 10000 pixels
#   3. MiniBatchKMeans fitted on all pixels
# Quality is measured with the mean squared error (MSE) and PSNR against
# the original image. A higher PSNR means the image is closer to the original.

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image
from sklearn.cluster import KMeans, MiniBatchKMeans

# Load a sample photo that comes with scikit-learn and scale to 0 to 1
image = load_sample_image("china.jpg") / 255.0
height, width, _ = image.shape
pixels = image.reshape(-1, 3)
print(f"Image: {width} x {height} pixels, unique colors: {len(np.unique((image * 255).astype(int).reshape(-1, 3), axis=0))}")
print()

def quantize(model, fit_data):
    start = time.time()
    model.fit(fit_data)
    labels = model.predict(pixels)
    elapsed = time.time() - start
    result = model.cluster_centers_[labels].reshape(height, width, 3)
    mse = np.mean((image - result) ** 2)
    psnr = 10 * np.log10(1.0 / mse)
    return result, elapsed, mse, psnr

rng = np.random.RandomState(42)
sample = pixels[rng.choice(len(pixels), 10000, replace=False)]

print(f"{'Colors':>6s} {'Method':26s} {'Time (s)':>9s} {'MSE':>9s} {'PSNR (dB)':>10s}")
images = {}
times = {}
for k in [4, 16, 64]:
    methods = {
        "KMeans (all pixels)": (KMeans(n_clusters=k, n_init=1, random_state=42), pixels),
        "KMeans (10000 sample)": (KMeans(n_clusters=k, n_init=1, random_state=42), sample),
        "MiniBatchKMeans": (MiniBatchKMeans(n_clusters=k, n_init=1, batch_size=2048, random_state=42), pixels),
    }
    for name, (model, data) in methods.items():
        result, elapsed, mse, psnr = quantize(model, data)
        images[(k, name)] = result
        times[(k, name)] = elapsed
        print(f"{k:6d} {name:26s} {elapsed:9.2f} {mse:9.5f} {psnr:10.2f}")
    print()

# Storage needed: 24 bits per pixel originally, log2(k) bits per pixel after
print("Approximate storage per pixel")
print("Original: 24 bits")
for k in [4, 16, 64]:
    bits = np.log2(k)
    print(f"{k} colors: {bits:.0f} bits, which is {24 / bits:.0f} times smaller (palette size ignored)")

# Show the original and the quantized images
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image)
axes[0].set_title("Original")
for ax, k in zip(axes[1:], [4, 16, 64]):
    ax.imshow(images[(k, "KMeans (10000 sample)")])
    ax.set_title(f"{k} colors")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Summary
print()
print("Summary:")
print("More colors give a lower error, but need more bits per pixel.")
full_time = times[(64, "KMeans (all pixels)")]
sample_time = times[(64, "KMeans (10000 sample)")]
mini_time = times[(64, "MiniBatchKMeans")]
print(f"At 64 colors, KMeans on all pixels took {full_time:.2f} s, on a 10000 pixel sample "
      f"{sample_time:.2f} s, and MiniBatchKMeans {mini_time:.2f} s.")
faster = [k for k in [4, 16, 64]
          if times[(k, "MiniBatchKMeans")] < times[(k, "KMeans (all pixels)")]]
print("MiniBatchKMeans was faster than full KMeans for these palette sizes:", faster)
print("Fitting on a small sample of pixels is the fastest option, and the PSNR is")
print("only slightly lower than fitting on all pixels.")
