


import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR, Imath
import numpy as np
import sys
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from .Points import PointsSource
from .Images import Image2D


from Util.Backend import backend as bd
from Util.Globals import (
    ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE,
    KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, ORIGIN, RNG
)
from Util.Misc import Magnitude, ArrayRotate, PolarToCartesian, RectPath
from Raytracing.RayBatch import RayBatch




def read_exr_channel(path, channel_name):
    """
    Read a single float32 channel from an EXR file as a 2D NumPy array.
    """
    exr = OpenEXR.InputFile(path)
    header = exr.header()
    dw = header["dataWindow"]
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = list(header["channels"].keys())

    # Try exact match first
    if channel_name not in channels:
        # If no exact match, try substring match (e.g. "Z" vs "Z.R" / "depth.Z")
        matches = [c for c in channels if channel_name in c]
        if not matches:
            raise ValueError(
                f"Channel '{channel_name}' not found. Available: {channels}"
            )
        channel_name = matches[0]  # pick the first match

    raw = exr.channel(channel_name, FLOAT)
    arr = np.frombuffer(raw, dtype=np.float32)
    arr = arr.reshape(height, width)

    return arr, channels, channel_name


def show_exr_channel(path, channel_name="Z"):
    """
    Read an EXR file and display one channel (e.g. 'Z' or 'A' / 'alpha') in a plot.
    """
    data, channels, chosen = read_exr_channel(path, channel_name)

    print(f"Available channels: {channels}")
    print(f"Displaying channel: {chosen}")
    print(f"Data range: min={data.min()}, max={data.max()}")

    plt.figure()
    plt.imshow(data, origin="lower")
    plt.title(f"{path} – {chosen}")
    plt.colorbar(label=chosen)
    plt.tight_layout()
    plt.show()


def make_quad_image(arrays):
    """
    Given a list of 4 2D arrays of identical shape, return a single 2×2 mosaic.
    Layout:
        [0] [1]
        [2] [3]
    All values are left as-is; global min/max will be used when displaying.
    """
    if len(arrays) != 4:
        raise ValueError("make_quad_image expects exactly 4 arrays.")

    h, w = arrays[0].shape
    for a in arrays[1:]:
        if a.shape != (h, w):
            raise ValueError("All arrays must have the same shape.")

    quad = np.zeros((2 * h, 2 * w), dtype=arrays[0].dtype)

    quad[0:h, 0:w] = arrays[0]  # top-left
    quad[0:h, w:2 * w] = arrays[1]  # top-right
    quad[h:2 * h, 0:w] = arrays[2]  # bottom-left
    quad[h:2 * h, w:2 * w] = arrays[3]  # bottom-right

    return quad


def show_exr_channel_quad(paths, channel_name="Z", labels=None):
    """
    Read the same channel from 4 EXR files, merge into a 2×2 mosaic,
    and display as a single image with a shared color scale.

    :param paths: list/tuple of 4 EXR file paths
    :param channel_name: channel to read (e.g. 'Z', 'A', 'alpha')
    :param labels: optional list of 4 strings to annotate the tiles
    """
    if len(paths) != 4:
        raise ValueError("show_exr_channel_quad expects exactly 4 paths.")

    arrays = []
    chosen_channels = []
    for p in paths:
        arr, _channels, chosen = read_exr_channel(p, channel_name)
        arrays.append(arr)
        chosen_channels.append(chosen)

    # Build one big mosaic
    quad = make_quad_image(arrays) * 10

    # Single global min/max so colors are comparable
    vmin = quad.min()
    vmax = quad.max()
    print(f"Global data range (all 4): min={vmin}, max={vmax}")
    print(f"Chosen channels: {chosen_channels}")

    plt.figure(figsize=(8, 8))
    im = plt.imshow(quad, origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=channel_name)

    # Optional simple annotations to remind which tile is which
    if labels is None:
        labels = [f"Img {i}" for i in range(4)]
    if len(labels) == 4:
        h, w = arrays[0].shape
        positions = [
            (w * 0.05, h * 0.05),                    # top-left
            (w * 1.05, h * 0.05),                    # top-right
            (w * 0.05, h * 1.05),                    # bottom-left
            (w * 1.05, h * 1.05),                    # bottom-right
        ]
        for (x, y), text in zip(positions, labels):
            plt.text(x, y, text, color="white",
                     fontsize=10, ha="left", va="bottom")

    plt.title(f"{channel_name} comparison (2×2 mosaic)")
    plt.tight_layout()
    plt.show()


def Verify():
    exr_path =  RectPath(r"resources/DepthSceneBG.exr")
    exr_path =  RectPath(r"resources/DepthSceneMG2.exr")
    #exr_path = RectPath(r"resources/DepthSceneFG.exr")
    #exr_path = RectPath(r"resources/DepthSceneMG.exr")
    # For depth:
    show_exr_channel_quad(
        [
            RectPath(r"resources/DepthSceneBG.exr"),
            RectPath(r"resources/DepthSceneMG2.exr"),
            RectPath(r"resources/DepthSceneFG.exr"),
            RectPath(r"resources/DepthSceneMG.exr")
        ]
    )
    # show_exr_channel(exr_path, "Z.R")


if __name__ == "__main__":
    # Example usage:
    #   python minimal_exr_viewer.py
    #
    # Change this to your file name and channel:
    exr_path = RectPath(r"resources/DepthSceneBG.exr")
    # For depth:
    show_exr_channel(exr_path, "Z.R")
    # For alpha (depending on how it’s named in your EXR, e.g. "A" or "alpha"):
    # show_exr_channel(exr_path, "A")
