import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Util.Backend as Backend
Backend.set_backend("CPU")
from Lens import Lens
from Surfaces.Stop import Stop
from Surfaces.Surface import Surface
from Raytracing.RayBatch import RayBatch


def rays(x, z=-1, dz=1, index=-1, y=0):
    data = np.zeros((len(x), 12))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 2] = z
    data[:, 5] = dz
    data[:, 6] = 550
    data[:, 7:9] = 1
    data[:, 10] = index
    return RayBatch(data)


def lens_with_stop():
    lens = Lens()
    lens.surfaces = [Surface(np.inf, 1, 10, "BAF9"), Stop(1),
                     Surface(np.inf, 1, 10, "BAF9")]
    for i, surface in enumerate(lens.surfaces):
        surface.SetCumulative(float(i))
    lens.stopIndex = 1
    lens._lastSurfaceIndex = 2
    lens.surfaces[1].EnforceSemiDiameter(2)
    lens.surfaces[1].apertureReferenceSemiDiameter = 2
    lens.entrancePupil.SetSamplePoints(np.array([[0., 0., 0.], [0., 4., 0.]]))
    lens.focalLength = 32.
    return lens


class StopTests(unittest.TestCase):
    def test_mask_both_directions_and_payload(self):
        stop = Stop(1)
        stop.SetCumulative(0.)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        stop.SetApertureShape(mask, 2.)
        for dz in (1, -1):
            incoming = rays([0., 1., 3.], z=-dz, dz=dz)
            result, tir, vig, stray = stop.Trace(incoming, np.ones(3), inverted=dz < 0)
            np.testing.assert_array_equal(vig, [False, True, True])
            np.testing.assert_array_equal(result.value[:, 3:], incoming.value[:1, 3:])
            np.testing.assert_array_equal(result.Position(), [[0, 0, 0]])
            self.assertEqual(tir.shape, (1,))
            self.assertTrue(stray.IsNoneType())
            self.assertEqual(incoming.value.shape[0], 3)

    def test_rgb_orientation_and_mapping(self):
        stop = Stop(1)
        stop.SetCumulative(0.)
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[1, 3] = 255
        stop.SetApertureShape(image, 2.)
        result, _, vig, _ = stop.Trace(rays([.8, -.8], y=.8), np.ones(2))
        np.testing.assert_array_equal(vig, [False, True])
        self.assertEqual(len(result.value), 1)

    def test_circle_empty_parallel_and_coplanar(self):
        stop = Stop(1)
        stop.SetCumulative(0.)
        stop.EnforceSemiDiameter(1.)
        result, _, vig, _ = stop.Trace(rays([0., 2.]), np.ones(2))
        np.testing.assert_array_equal(vig, [False, True])
        for incoming in (rays([]), rays([0.], dz=0), rays([0.], z=1)):
            result, _, _, _ = stop.Trace(incoming, np.ones(len(incoming.value)))
            self.assertTrue(result.IsNoneType())
        result, _, _, _ = stop.Trace(rays([0.], z=0), np.ones(1))
        self.assertEqual(len(result.value), 1)

    def test_sequential_records_stop_and_handles_all_blocked(self):
        for reflection in (False, True):
            lens = lens_with_stop()
            incoming = rays([0., 3.])
            incoming.value[:, 4] = .01
            result, path, stray = lens.Propagate(incoming, recordPath=True, reflection=reflection)
            self.assertEqual(len(result.value), 1)
            self.assertEqual(len(path.position), 4)
            np.testing.assert_array_equal(path.vignetted[2], [False, True])
            incoming = rays([3.])
            incoming.value[:, 4] = .01
            result, _, _ = lens.Propagate(incoming, reflection=reflection)
            self.assertTrue(result.IsNoneType())

    def test_nonsequential_stop_crossings(self):
        for dz, z, index in ((-1, 2., 1), (1, 0., 0)):
            lens = lens_with_stop()
            stop = lens.surfaces[1]
            mask = np.zeros((5, 5), dtype=bool)
            mask[2, 2] = True
            stop.SetApertureShape(mask, 2.)
            incoming = rays([0., 1.], z=z, dz=dz, index=index)
            incoming.value[:, 4] = .01
            with patch.object(stop, "Trace", wraps=stop.Trace) as trace:
                result, _ = lens._BounceReflectionAlt(incoming)
            self.assertTrue(trace.called)
            self.assertFalse(result.IsNoneType())
            np.testing.assert_array_equal(result.Position()[:, 0], [0.])
            np.testing.assert_array_equal(result.SurfaceIndex(), [2])
            self.assertTrue(np.all(np.isfinite(result.value)))

    def test_backward_blade_hit_has_no_exit(self):
        lens = lens_with_stop()
        lens.surfaces[1].SetApertureShape(np.zeros((5, 5), dtype=bool), 2.)
        result, remaining = lens._BounceReflectionAlt(rays([0.], z=2., dz=-1, index=1))
        self.assertTrue(result.IsNoneType())
        self.assertTrue(remaining.IsNoneType())

    def test_primary_reflections_keep_space_index_after_stop(self):
        lens = lens_with_stop()
        incoming = rays([0.])
        incoming.value[:, 4] = .01
        with patch.object(lens, "_BounceReflectionAlt", wraps=lens._BounceReflectionAlt) as bounce:
            _, _, ghosts = lens.Propagate(incoming, reflection=True, iteCount=1)
        self.assertTrue(bounce.called)
        reflected = bounce.call_args.args[0]
        self.assertIn(1, reflected.SurfaceIndex())
        self.assertFalse(ghosts.IsNoneType())
        self.assertTrue(np.all(np.isfinite(ghosts.value)))

    def test_through_helper_culls_at_stop(self):
        lens = lens_with_stop()
        result = lens._PropagateReflectedThrough(rays([0., 3.], z=0., index=0))
        np.testing.assert_array_equal(result.Position()[:, 0], [0.])

    def test_aperture_switching_and_full_open_reset(self):
        lens = lens_with_stop()
        stop = lens.surfaces[1]
        lens.SetAperture(8, useDiaphragm=False)
        self.assertAlmostEqual(stop.clearSemiDiameter, 1.)
        self.assertAlmostEqual(lens.entrancePupil.clearSemiDiameter, 2.)
        lens.SetAperture(8)
        np.testing.assert_array_equal(stop.apertureShape, lens.entrancePupil._alphaShape)
        self.assertAlmostEqual(stop.apertureShapeSemiDiameter, 2.)
        stopped = np.count_nonzero(stop.apertureShape[..., :3].mean(axis=-1) > 0.5)
        lens.SetAperture(4)
        opened = np.count_nonzero(stop.apertureShape[..., :3].mean(axis=-1) > 0.5)
        self.assertGreater(opened, stopped)
        lens.SetAperture(8, useDiaphragm=False)
        self.assertIsNone(stop.apertureShape)
        self.assertIsNone(lens.entrancePupil._alphaShape)
        self.assertAlmostEqual(stop.clearSemiDiameter, 1.)


if __name__ == "__main__":
    unittest.main()
