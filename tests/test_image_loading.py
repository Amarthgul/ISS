import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import Util.Backend as Backend

Backend.set_backend("CPU")

from ObjectSpace.Image2DFlat import Image2DFlat
from ObjectSpace.ImageExt import (
    Image2DFlatHighlightExtension,
    Image2DVariHighlightExtension,
)
from ObjectSpace.Images import Image2D
from ObjectSpace.ImageVariDepth import Image2DVariDepth


class ImageLoadingTests(unittest.TestCase):
    def setUp(self):
        self.tempDirectory = tempfile.TemporaryDirectory()
        self.tempPath = Path(self.tempDirectory.name)

        rgba = np.zeros((2, 4, 4), dtype=np.uint8)
        rgba[..., :3] = 255
        rgba[..., 3] = np.array([
            [255, 255, 0, 0],
            [255, 255, 0, 0],
        ])
        self.rgbPath = self.tempPath / "rgb.png"
        Image.fromarray(rgba).save(self.rgbPath)

        depth = np.zeros((2, 4, 4), dtype=np.uint8)
        depth[..., :3] = 128
        depth[..., 3] = np.array([
            [255, 0, 255, 0],
            [255, 0, 255, 0],
        ])
        self.depthPath = self.tempPath / "depth.png"
        Image.fromarray(depth).save(self.depthPath)

    def tearDown(self):
        self.tempDirectory.cleanup()

    def test_shared_loader_ownership(self):
        self.assertIs(Image2DFlat.LoadFrom8bitRGB, Image2D.LoadFrom8bitRGB)
        self.assertIs(Image2DVariDepth.LoadFrom8bitRGB, Image2D.LoadFrom8bitRGB)
        self.assertIs(
            Image2DVariHighlightExtension.LoadFrom8bitZ,
            Image2DVariDepth.LoadFrom8bitZ,
        )

    def test_flat_loader_preserves_alpha_and_builds_points(self):
        image = Image2DFlat().LoadFrom8bitRGB(self.rgbPath)

        self.assertEqual(image.rgbArray.shape, (2, 4, 3))
        self.assertEqual(image.alphaArray.shape, (2, 4))
        self.assertEqual(image.pointSource.value.shape, (8, 6))
        self.assertEqual(
            np.count_nonzero(np.any(image.pointSource.value[:, 3:6] > 0, axis=1)),
            4,
        )

    def test_vari_depth_loader_merges_alpha_and_builds_points(self):
        image = Image2DVariDepth()
        image.LoadFrom8bit(self.rgbPath, self.depthPath)

        self.assertEqual(image.zArray.shape, (2, 4))
        self.assertEqual(np.count_nonzero(image.alphaArray), 2)
        self.assertEqual(image.pointSource.value.shape, (8, 6))
        self.assertEqual(
            np.count_nonzero(np.any(image.pointSource.value[:, 3:6] > 0, axis=1)),
            2,
        )

    def test_flat_highlight_delegates_loading_and_floods_depth(self):
        image = Image2DFlatHighlightExtension()
        image.zDistance = 1234
        image.LoadFrom8bitRGB(self.rgbPath)

        np.testing.assert_allclose(image.zArray, 1234)
        self.assertEqual(float(image.zDistance), -1234)
        self.assertEqual(image.pointSource.value.shape, (8, 6))
        np.testing.assert_allclose(image.rgbArray[:, 2:, :], 0)

    def test_sparse_layer_keeps_full_sampling_population(self):
        dense = Image2DVariDepth()
        sparse = Image2DVariDepth()

        for image in (dense, sparse):
            image.rgbArray = np.ones((4, 4, 3))
            image.zArray = np.ones((4, 4))
            image.zDistance = -image.zArray

        dense.alphaArray = np.ones((4, 4))
        sparse.alphaArray = np.zeros((4, 4))
        sparse.alphaArray[0, :] = 1

        dense.Refresh()
        sparse.Refresh()

        self.assertEqual(dense.pointSource.value.shape[0], 16)
        self.assertEqual(sparse.pointSource.value.shape[0], 16)
        self.assertEqual(
            np.count_nonzero(np.any(sparse.pointSource.value[:, 3:6] > 0, axis=1)),
            4,
        )

        def valid_samples_over_full_cycle(pointSource, sampleCount=4):
            validSamples = 0
            for _ in range(4):
                if pointSource.sampleRecord.shape[0] <= sampleCount:
                    selected = pointSource.value
                else:
                    indices = pointSource._SelectLeastSampled(sampleCount)
                    pointSource.sampleRecord[indices] += 1
                    selected = pointSource.value[indices]
                validSamples += np.count_nonzero(
                    np.any(selected[:, 3:6] > 0, axis=1)
                )
            return validSamples

        self.assertEqual(valid_samples_over_full_cycle(dense.pointSource), 16)
        self.assertEqual(valid_samples_over_full_cycle(sparse.pointSource), 4)

    def test_exr_reader_is_shared_by_flat_and_vari_depth(self):
        exrPath = REPO_ROOT / "resources" / "allChannels.exr"

        flat = Image2DFlat()
        flat.imageDimensionOverride = 8
        flat.LoadFromEXR(exrPath)

        vari = Image2DVariDepth()
        vari.imageDimensionOverride = 8
        vari.LoadFromEXR(exrPath)

        self.assertEqual(flat.rgbArray.shape, (3, 8, 3))
        self.assertIsNotNone(flat.pointSource)
        self.assertEqual(vari.zArray.shape, (3, 8))
        self.assertEqual(vari.GetAOVNames()[:2], ["A", "depth.Z"])


if __name__ == "__main__":
    unittest.main()
