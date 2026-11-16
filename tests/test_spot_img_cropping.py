import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import OpenEXR
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from Util.SpotImgCropping import CropSpotImage, CropSpotImages, RunSpotCropping, SpotGridInfo


class SpotImgCroppingTests(unittest.TestCase):
    def test_angular_sampling_and_direction_selection(self):
        grid = SpotGridInfo(50, angleOfView=(60, 40), ratio=0.9)
        expected = np.array([
            [50 * math.tan(math.radians(x)), 50 * math.tan(math.radians(y))]
            for x, y in ((0, 0), (6.75, 4.5), (13.5, 9), (20.25, 13.5), (27, 18))
        ])
        np.testing.assert_allclose(grid.Centers("diagonal"), expected)
        np.testing.assert_allclose(grid.Centers("horizontal"), expected * (1, 0))
        np.testing.assert_allclose(grid.Centers("vertical"), expected * (0, 1))

    def test_derived_field_of_view_reaches_imager_edges(self):
        grid = SpotGridInfo(50, ratio=1)
        np.testing.assert_allclose(grid.Centers("diagonal")[-1], (18, 12))
        explicit = SpotGridInfo(50, angleOfView=(
            math.degrees(2 * math.atan(18 / 50)),
            math.degrees(2 * math.atan(12 / 50)),
        ))
        np.testing.assert_allclose(
            SpotGridInfo(50).Centers("diagonal"), explicit.Centers("diagonal")
        )

    def test_axis_crops_and_padding_preserve_hdr_pixels(self):
        grid = SpotGridInfo(1, sample=3, ratio=1, angleOfView=(90, 90), imagerSize=(4, 4))
        image = np.arange(40 * 40 * 3, dtype=np.float32).reshape(40, 40, 3) - 1000
        strip = CropSpotImage(image, grid, boxSize=(0.4, 0.6), padding=0.2)
        self.assertEqual(strip.shape, (6, 10, 3))
        np.testing.assert_array_equal(strip[:, :4], image[17:23, 18:22])
        np.testing.assert_array_equal(strip[:, 4:6], 0)
        np.testing.assert_array_equal(strip[:, 6:], image[17:23, 28:32])

        strip = CropSpotImage(image, grid, direction="vertical", layout="vertical",
                              boxSize=(0.4, 0.6), padding=0.2)
        self.assertEqual(strip.shape, (10, 6, 3))
        np.testing.assert_array_equal(strip[:4], image[18:22, 17:23])
        np.testing.assert_array_equal(strip[4:6], 0)
        np.testing.assert_array_equal(strip[6:], image[28:32, 17:23])

    def test_diagonal_rotation_with_unequal_pixel_densities(self):
        # The corner is (4, 3) mm, so the radial unit vector is (0.8, 0.6).
        # Use different x/y densities to expose rotations done in pixel space.
        grid = SpotGridInfo(1, sample=3, ratio=1, imagerSize=(10, 8), angleOfView=(
            math.degrees(2 * math.atan(4)), math.degrees(2 * math.atan(3)),
        ))
        x, y = np.meshgrid((np.arange(400) + 0.5) / 40 - 5,
                           (np.arange(160) + 0.5) / 20 - 4)
        radialRamp = (0.8 * x + 0.6 * y).astype(np.float32)
        for layout in ("horizontal", "vertical"):
            with self.subTest(layout=layout):
                strip = CropSpotImage(radialRamp, grid, direction="diagonal", layout=layout,
                                      boxSize=(1, 0.5), padding=0)
                if layout == "horizontal":
                    expected = np.broadcast_to((np.arange(40) - 19.5) / 40, (10, 40))
                    np.testing.assert_allclose(strip[:, :40], expected, atol=0.002)
                    np.testing.assert_allclose(strip[:, 40:], expected + 5, atol=0.002)
                else:
                    expected = np.broadcast_to(((np.arange(20) - 9.5) / 20)[:, None], (20, 20))
                    np.testing.assert_allclose(strip[:20], expected, atol=0.002)
                    np.testing.assert_allclose(strip[20:], expected + 5, atol=0.002)

    def test_outside_imager_is_black_with_fixed_box_size(self):
        grid = SpotGridInfo(1, sample=3, ratio=1, imagerSize=(2, 2))
        strip = CropSpotImage(np.full((20, 20), 7, dtype=np.float32), grid,
                              boxSize=(0.4, 0.4), padding=0)
        self.assertEqual(strip.shape, (4, 8))
        np.testing.assert_array_equal(strip[:, :6], 7)
        np.testing.assert_array_equal(strip[:, 6:], 0)

    def test_nine_by_nine_grid_contains_five_spots_in_order(self):
        # Paint isolated disks at independently calculated ideal positions.
        image = np.zeros((240, 360), dtype=np.float32)
        y, x = np.indices(image.shape)
        for index, angle in enumerate((0, 4.5, 9, 13.5, 18)):
            centerX = (50 * math.tan(math.radians(angle)) + 18) * 10 - 0.5
            image[(x - centerX) ** 2 + (y - 119.5) ** 2 < 4] = index + 1
        grid = SpotGridInfo(50, angleOfView=(40, 26), ratio=0.9)
        strip = CropSpotImage(image, grid, boxSize=(1, 1), padding=0)
        self.assertEqual(strip.shape, (10, 50))
        for index in range(5):
            self.assertEqual(strip[:, index * 10:(index + 1) * 10].max(), index + 1)

    def test_exr_batch_preserves_channels_hdr_and_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            channels = {
                name: np.full((40, 40), value, dtype=np.float32)
                for name, value in (("R", 4), ("G", -2), ("B", 0.5), ("A", 0.25), ("depth.Z", 42))
            }
            sourceBytes = {}
            for name in ("SonnarZ0.0.exr", "SonnarZ0.1.EXR", "Unrelated.exr"):
                path = folder / name
                with OpenEXR.File({"type": OpenEXR.scanlineimage}, channels.copy()) as image:
                    image.write(str(path))
                sourceBytes[name] = path.read_bytes()
            grid = SpotGridInfo(1, sample=3, ratio=1, angleOfView=(90, 90), imagerSize=(4, 4))
            outputs = CropSpotImages(folder, r"SonnarZ0\.\d\.exr", grid,
                                     boxSize=(0.4, 0.4), padding=0.2)
            self.assertEqual([path.name for path in outputs], [
                "SonnarZ0.0_spots_horizontal_horizontal.exr",
                "SonnarZ0.1_spots_horizontal_horizontal.EXR",
            ])
            for path in outputs:
                with OpenEXR.File(str(path), separate_channels=True) as image:
                    self.assertEqual(set(image.channels()), set(channels))
                    for name, channel in image.channels().items():
                        self.assertEqual(channel.pixels.dtype, np.float32)
                        self.assertEqual(channel.pixels.shape, (4, 10))
                        np.testing.assert_array_equal(channel.pixels[:, :4], channels[name][0, 0])
                        np.testing.assert_array_equal(channel.pixels[:, 4:6], 0)
                        np.testing.assert_array_equal(channel.pixels[:, 6:], channels[name][0, 0])
            for name, content in sourceBytes.items():
                self.assertEqual((folder / name).read_bytes(), content)
            self.assertEqual(len(CropSpotImages(folder, r".*\.exr", grid)), 3)
            self.assertEqual(CropSpotImages(folder, r"Missing\.exr", grid), [])

    def test_png_keeps_format_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pixels = np.full((40, 40, 4), (200, 100, 50, 128), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / "spots.png")
            outputs = RunSpotCropping(
                folder=folder, pattern=r"spots\.png", focalLength=1, sample=3,
                ratio=1, angleOfView=(90, 90), imagerSize=(4, 4),
                boxSize=(0.4, 0.4), padding=0,
            )
            with Image.open(outputs[0]) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.size, (8, 4))
                np.testing.assert_array_equal(np.array(image), np.broadcast_to(pixels[0, 0], (4, 8, 4)))


if __name__ == "__main__":
    unittest.main()
