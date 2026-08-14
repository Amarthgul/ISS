import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import Util.Backend as Backend

Backend.set_backend("CPU")

from ObjectSpace.ImageStack import ExampleStackSpotGrid, ImageStack
from ObjectSpace.ImageVariDepth import Image2DVariDepth
from ObjectSpace.Points import PointsSource


class ExampleStackSpotGridTests(unittest.TestCase):
    def test_selects_next_wider_focal_length_scene(self):
        cases = (
            (19, 105),
            (20, 105),
            (21, 50),
            (36, 50),
            (40, 50),
            (41, 35),
            (54, 35),
            (55, 28),
            (65, 28),
            (66, 24),
            (74, 24),
            (75, 15),
            (100, 15),
            (101, 15),
        )

        with (
            patch.object(PointsSource, "GenerateGridSpots"),
            patch.object(Image2DVariDepth, "LoadFromEXR") as loadFromEXR,
            patch.object(ImageStack, "PrintLayerTags"),
        ):
            for horizontalAoV, expectedFocalLength in cases:
                with self.subTest(horizontalAoV=horizontalAoV):
                    loadFromEXR.reset_mock()
                    stack = ExampleStackSpotGrid(horizontalAoV)

                    expectedPaths = [
                        f"resources/VarFocalScene/MG_FL{expectedFocalLength}.exr",
                        f"resources/VarFocalScene/FG_FL{expectedFocalLength}.exr",
                    ]
                    expectedLayers = ["gridPointSource", "MG", "FG"]
                    if expectedFocalLength != 15:
                        expectedPaths.append(
                            f"resources/VarFocalScene/AP_FL{expectedFocalLength}.exr"
                        )
                        expectedLayers.append("AP")

                    self.assertEqual(
                        [call.args[0] for call in loadFromEXR.call_args_list],
                        expectedPaths,
                    )
                    self.assertEqual(list(stack.images), expectedLayers)


if __name__ == "__main__":
    unittest.main()
