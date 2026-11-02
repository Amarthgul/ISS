"""Plotting boundary regressions for sampled pupil surfaces."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from Util import PltPlot


class DrawPupilTests(unittest.TestCase):
    def check_profile(self, radius, depth, expected_radius, expected_depth):
        axis = Mock()
        with patch.object(PltPlot, 'RENDER_MODE', False):
            PltPlot.DrawPupil(radius, depth, num_points=9,
                             surfaceColor='red', ax=axis)
        x, y, z = axis.plot_surface.call_args.args
        shape = (9, len(expected_radius))
        self.assertEqual(x.shape, shape)
        self.assertEqual(y.shape, shape)
        self.assertEqual(z.shape, shape)
        np.testing.assert_allclose(np.hypot(x, y),
                                   np.broadcast_to(expected_radius, shape))
        np.testing.assert_allclose(z, np.broadcast_to(expected_depth, shape))
        self.assertEqual(axis.plot_surface.call_args.kwargs['color'], 'red')

    def test_sampled_profile_preserves_each_depth(self):
        self.check_profile([0., 2., 4.], [10., 11., 13.],
                           [0., 2., 4.], [10., 11., 13.])

    def test_scalar_and_singleton_compatibility(self):
        self.check_profile(4., 10., [4.], [10.])
        self.check_profile(np.array([4.]), np.array([10.]), [4.], [10.])
        self.check_profile([0., 2., 4.], 10., [0., 2., 4.], [10., 10., 10.])

    def test_gpu_profiles_convert_at_plot_boundary(self):
        class DeviceArray:
            def __init__(self, values):
                self.values = np.asarray(values)

            def get(self):
                return self.values

        with patch.object(PltPlot, 'backend_name', 'cupy'):
            self.check_profile(DeviceArray([0., 2., 4.]),
                               DeviceArray([10., 11., 13.]),
                               [0., 2., 4.], [10., 11., 13.])


if __name__ == '__main__':
    unittest.main()
