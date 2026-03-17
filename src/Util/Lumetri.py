


"""
Relevant visualization package. Not used in the project directly.
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def compute_luma(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def upsample_image(rgb, factor):
    if factor <= 1:
        return rgb

    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def plot_waveform(ax, rgb, gain):
    h, w, _ = rgb.shape
    luma = compute_luma(rgb)

    x = np.tile(np.arange(w), h)
    y = luma.reshape(-1)

    ax.scatter(
        x,
        y,
        s=0.2 * gain,
        alpha=min(0.02 * gain, 1.0),
        c="white"
    )

    ax.set_xlim(0, w)
    ax.set_ylim(0, 1)
    ax.set_title("Luma Waveform", color="white")
    ax.tick_params(colors="white")


def plot_rgb_parade(ax, rgb, gain):
    h, w, _ = rgb.shape
    channels = [rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    colors = ["red", "green", "blue"]

    for i, ch in enumerate(channels):
        x = np.tile(np.arange(w), h) + i * w
        y = ch.reshape(-1)

        ax.scatter(
            x,
            y,
            s=0.2 * gain,
            alpha=min(0.02 * gain, 1.0),
            c=colors[i]
        )

    ax.set_xlim(0, w * 3)
    ax.set_ylim(0, 1)
    ax.set_title("RGB Parade", color="white")
    ax.tick_params(colors="white")


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = (b - y) / 1.8556
    cr = (r - y) / 1.5748

    return cb, cr


def plot_vectorscope(ax, rgb, gain):
    cb, cr = rgb_to_ycbcr(rgb)

    x = cr.flatten()
    y = cb.flatten()

    ax.scatter(
        x,
        y,
        s=0.3 * gain,
        alpha=min(0.03 * gain, 1.0),
        c="cyan"
    )

    ax.set_xlim(-0.7, 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")

    ax.set_title("Vectorscope", color="white")
    ax.tick_params(colors="white")

    circle = plt.Circle((0, 0), 0.5, fill=False, color="white", alpha=0.3)
    ax.add_artist(circle)


def apply_scope_style(ax):
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")


def main() -> None:

    image_path = None

    rgb = read_image(str(image_path))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Lumetri-style Scopes: {image_path}", fontsize=14)

    # Original image
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Image")
    axes[0, 0].axis("off")

    gain = 20

    # Luma waveform
    plot_waveform(axes[0, 1], rgb, gain)

    # RGB parade
    plot_rgb_parade(axes[1, 0], rgb, gain)

    # Vectorscope
    plot_vectorscope(axes[1, 1], rgb, gain)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()