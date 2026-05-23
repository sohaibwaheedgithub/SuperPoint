import matplotlib.pyplot as plt
import numpy as np

# Example: create 64 random images of size 240x320
# Replace this with your actual images
images = [
    np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    for _ in range(64)
]

fig, axes = plt.subplots(
    8, 8,
    figsize=(32, 24),   # 320*8 / 100 , 240*8 / 100
    gridspec_kw={'wspace': 0.05, 'hspace': 0.05}
)

for idx, ax in enumerate(axes.flat):
    ax.imshow(images[idx])

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add visible border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor("white")

plt.tight_layout()