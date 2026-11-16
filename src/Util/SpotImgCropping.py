"""Extract center-to-edge strips from ImagingSystem.SpotGrid renders.

Run from the repository root, for example (SonnarZ uses ratio=0.9)::

    python src/Util/SpotImgCropping.py --focal-length 49.97645 --ratio 0.9
    python src/Util/SpotImgCropping.py --direction diagonal --layout vertical

From another script in the repository root::

    from src.Util.SpotImgCropping import RunSpotCropping

    outputs = RunSpotCropping(
        focalLength=49.97645, ratio=0.9, direction="diagonal",
        layout="horizontal", boxSize=(3.0, 3.0), padding=0.25,
    )

The filename regex matches whole filenames. Each match produces one strip in
the input folder's SpotCrops subfolder, retaining the input file extension.
EXR files retain every channel as float32, without exposure or gamma changes.
Other formats are read and written through Pillow in their original mode.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Literal

import cv2
import numpy as np
import OpenEXR
from PIL import Image


# Edit these defaults, or override them on the command line.
IMAGE_FOLDER = Path(__file__).resolve().parents[2] / "resources" / "Results"
IMAGE_PATTERN = r"SonnarZ\d+(?:\.\d+)?(?:p)?\.exr"

Direction = Literal["horizontal", "vertical", "diagonal"]
Layout = Literal["horizontal", "vertical"]


@dataclass
class SpotGridInfo:
    """Ideal rectilinear geometry matching SpotGrid and GenerateGridSpots.

    focalLength and imagerSize (width, height) are in mm. sample is the odd
    number of spots on EACH axis, at least 3, as in SpotGrid(sample=...).
    angleOfView is the FULL (horizontal, vertical) lens field in degrees,
    BEFORE applying SpotGrid's ratio. If omitted, derive it from focalLength
    and imagerSize. For a lens object, pass lens.GetAoV(halfAngle=False)[:2].

    SpotGrid currently calls GetAoV with its default 36x24 mm gate, even for
    other imagers. For those renders, pass angleOfView explicitly and set
    imagerSize to the actual gate. Crop centers are ideal f*tan(theta), not
    measured spot centroids; distortion and defocus can shift real spots.
    """

    focalLength: float = 50.0
    sample: int = 9
    ratio: float = 0.92
    angleOfView: tuple[float, float] | None = None
    imagerSize: tuple[float, float] = (36.0, 24.0)

    def Centers(self, direction: Direction) -> np.ndarray:
        """Return center-to-edge (x, y) offsets in mm, right/down in the file."""
        if self.angleOfView is None:
            halfAngles = np.arctan(np.array(self.imagerSize) / (2 * self.focalLength))
        else:
            halfAngles = np.deg2rad(self.angleOfView) / 2

        axes = {"horizontal": (1, 0), "vertical": (0, 1), "diagonal": (1, 1)}
        fractions = np.linspace(0, 1, self.sample // 2 + 1)
        angles = fractions[:, None] * halfAngles * self.ratio * axes[direction]
        return self.focalLength * np.tan(angles)


def CropSpotImage(
        image: np.ndarray,
        grid: SpotGridInfo,
        direction: Direction = "horizontal",
        layout: Layout = "horizontal",
        boxSize: tuple[float, float] = (2.0, 2.0),
        padding: float = 0.25) -> np.ndarray:
    """Crop a row-major image into one center-to-edge strip.

    boxSize is (radial length, tangential length) in mm; padding is the gap
    between boxes in mm, with no outer border. Lengths must give positive
    pixel dimensions and padding must be nonnegative. Input is a 2D channel
    or an HxWxC array. Sampling is bilinear in float32, with black outside the
    image. Pixel centers lie half a pixel inside the physical imager edges.

    The input raster covers the full imager. The chosen half is right, down,
    or lower-right. Each box follows the radial direction at its ideal center,
    then rotates that direction onto
    the output row/column. The central box uses the first off-axis direction.
    A diagonal on a rectangular imager is therefore not assumed to be 45 deg.
    Output pixel density along x/y matches the source imager's pixel density.
    """
    image = image.astype(np.float32, copy=False)
    pixelsPerMM = np.array(image.shape[1::-1]) / grid.imagerSize
    centers = grid.Centers(direction)
    angles = np.arctan2(centers[:, 1], centers[:, 0])
    angles[0] = angles[1]

    axis = {"horizontal": 1, "vertical": 0}[layout]
    sizeMM = boxSize if layout == "horizontal" else boxSize[::-1]
    width, height = np.rint(np.array(sizeMM) * pixelsPerMM).astype(int)
    gap = round(padding * pixelsPerMM[1 - axis])
    shape = [height, width, *image.shape[2:]]
    step = shape[axis] + gap
    shape[axis] = len(centers) * step - gap
    strip = np.zeros(shape, dtype=np.float32)
    cropCenter = (np.array([width, height]) - 1) / 2

    for index, (center, angle) in enumerate(zip(centers, angles)):
        angle -= np.pi / 2 if layout == "vertical" else 0
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        # Map output pixel centers directly into the original image, so each
        # spot is interpolated once and rotated corners are never pre-clipped.
        transform = rotation * pixelsPerMM[:, None] / pixelsPerMM[None, :]
        centerPx = (center + np.array(grid.imagerSize) / 2) * pixelsPerMM - 0.5
        offset = centerPx - transform @ cropCenter
        crop = cv2.warpAffine(
            image, np.column_stack((transform, offset)), (width, height),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        ).reshape(height, width, *image.shape[2:])
        start = index * step
        if layout == "horizontal":
            strip[:, start:start + width] = crop
        else:
            strip[start:start + height, :] = crop

    return strip


def CropSpotImages(
        folder: Path,
        pattern: str,
        grid: SpotGridInfo,
        direction: Direction = "horizontal",
        layout: Layout = "horizontal",
        boxSize: tuple[float, float] = (2.0, 2.0),
        padding: float = 0.25) -> list[Path]:
    """Process all regex-matching files, nonrecursively, in filename order.

    EXR input is a single flat part with full-resolution channels. Output EXRs
    use lossless ZIP compression and fresh windows sized to the strip. Other
    inputs are Pillow-supported grayscale/RGB/RGBA images. Existing strips of
    the same name are replaced; source files are never changed. An unmatched
    regex returns an empty list. Generated strips live in a subfolder so a
    later invocation cannot accidentally crop them again.
    """
    matcher = re.compile(pattern, re.IGNORECASE)
    sources = sorted(path for path in folder.iterdir()
                     if path.is_file() and matcher.fullmatch(path.name))
    outputs = []
    outputFolder = folder / "SpotCrops"
    outputFolder.mkdir(exist_ok=True)

    for source in sources:
        output = outputFolder / f"{source.stem}_spots_{direction}_{layout}{source.suffix}"
        if source.suffix.lower() == ".exr":
            with OpenEXR.File(str(source), separate_channels=True) as image:
                channels = {
                    name: CropSpotImage(channel.pixels, grid, direction, layout, boxSize, padding)
                    for name, channel in image.channels().items()
                }
            header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
            with OpenEXR.File(header, channels) as image:
                image.write(str(output))
        else:
            with Image.open(source) as image:
                pixels = np.array(image)
                strip = CropSpotImage(pixels, grid, direction, layout, boxSize, padding)
                Image.fromarray(strip.astype(pixels.dtype)).save(output)
        outputs.append(output)

    return outputs


def RunSpotCropping(
        *,
        folder: Path = IMAGE_FOLDER,
        pattern: str = IMAGE_PATTERN,
        focalLength: float = 50.0,
        sample: int = 9,
        ratio: float = 0.92,
        angleOfView: tuple[float, float] | None = None,
        imagerSize: tuple[float, float] = (36.0, 24.0),
        direction: Direction = "horizontal",
        layout: Layout = "horizontal",
        boxSize: tuple[float, float] = (2.0, 2.0),
        padding: float = 0.25) -> list[Path]:
    """Run a batch directly from Python using the command-line defaults.

    Pass folder as a pathlib.Path. All lengths are in mm; angleOfView is the
    full horizontal/vertical field in degrees before ratio is applied.
    See SpotGridInfo and CropSpotImage for geometry and box conventions.
    Returns the written paths without parsing arguments or printing output.
    """
    grid = SpotGridInfo(focalLength, sample, ratio, angleOfView, imagerSize)
    return CropSpotImages(folder, pattern, grid, direction, layout, boxSize, padding)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", type=Path, default=IMAGE_FOLDER)
    parser.add_argument("--pattern", default=IMAGE_PATTERN, help="Full filename regex")
    parser.add_argument("--focal-length", type=float, default=50.0, help="Focal length in mm")
    parser.add_argument("--sample", type=int, default=9, help="Odd spot count on each axis (>=3)")
    parser.add_argument("--ratio", type=float, default=0.92, help="SpotGrid field-angle ratio")
    parser.add_argument("--angle-of-view", type=float, nargs=2, metavar=("X", "Y"),
                        help="Full lens angles in degrees, before applying ratio")
    parser.add_argument("--imager-size", type=float, nargs=2, default=(36.0, 24.0),
                        metavar=("WIDTH", "HEIGHT"), help="Imager dimensions in mm")
    parser.add_argument("--direction", choices=("horizontal", "vertical", "diagonal"),
                        default="horizontal")
    parser.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal")
    parser.add_argument("--box-size", type=float, nargs=2, default=(2.0, 2.0),
                        metavar=("RADIAL", "TANGENTIAL"), help="Crop dimensions in mm")
    parser.add_argument("--padding", type=float, default=0.25, help="Gap between crops in mm")
    args = parser.parse_args()
    outputs = RunSpotCropping(
        folder=args.folder, pattern=args.pattern, focalLength=args.focal_length,
        sample=args.sample, ratio=args.ratio, angleOfView=args.angle_of_view,
        imagerSize=args.imager_size, direction=args.direction, layout=args.layout,
        boxSize=args.box_size, padding=args.padding,
    )
    for output in outputs:
        print(output)
    print(f"Saved {len(outputs)} spot strips.")


if __name__ == "__main__":
    main()
