"""Mixed-order surface plots retain readable rows and physical alignment."""

from io import BytesIO
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Util.Backend as Backend
Backend.set_backend("CPU")

from Lens import Lens
from Surfaces.Stop import Stop
from Surfaces.Surface import Surface
from Util.Analysis import SurfaceData as SD


def objective():
    lens = Lens()
    lens.surfaces = [Surface(50, 5, 10, "BAF9"), Surface(-50, 8, 10),
                     Stop(2), Surface(70, 4, 10, "BK7"), Surface(-60, 15, 10)]
    lens.groups = [[0, 1], [3, 4]]
    lens.lenses = [[0, 1], [3, 4]]
    lens.stopIndex = 2
    lens.entrancePupil.clearSemiDiameter = 3.0
    lens.surfaces[2].clearSemiDiameter = 3.0
    vertex = 0.0
    for surface in lens.surfaces:
        surface.SetCumulative(vertex)
        vertex += float(surface.thickness)
    lens.totalAxialLength = 19.0
    return lens


class SurfaceDataAberrationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_track_counts_preserve_values_and_default_plot_proportions(self):
        lens = objective()
        with patch.object(lens, "ComputeHigherOrder", wraps=lens.ComputeHigherOrder) as compute:
            baseline, baselineAxes = lens.PlotSurfaceData(PlotTrackLength=100.)
            compute.assert_not_called()
        np.testing.assert_allclose(baseline.get_size_inches(), [16., 10.])
        self.assertEqual([axis.get_ylabel() for axis in baselineAxes],
                         ["Height (mm)", "Group power", "nd", "Vd"])
        baselineTrackHeight = baselineAxes[1].get_position().height * baseline.get_figheight()
        previousHeight = baseline.get_figheight()
        selected = SD.displayConfigAberrations[3:] + [
            SD.SurfaceDataType.W171, SD.SurfaceDataType.W260,
            SD.SurfaceDataType.W262, SD.SurfaceDataType.W351,
        ]
        expected = lens.ComputeHigherOrder(referenceField=6., referenceFieldUnits="degrees")
        coefficients = [row.AsDict() for row in expected.CumulativeWavefrontCoefficients()]

        for count, order in [(5, 3), (8, 5), (10, 7), (14, 7)]:
            with self.subTest(count=count):
                tracks = selected[:count]
                with patch.object(lens, "ComputeHigherOrder", wraps=lens.ComputeHigherOrder) as compute:
                    fig, axes = lens.PlotSurfaceData(
                        DisplayConfig=SD.displayConfigAberrations[:3] + tracks,
                        PlotTrackLength=100., SeidelReferenceField=6.)
                    self.assertEqual(compute.call_count, 1)
                    self.assertEqual(compute.call_args.kwargs["maxOrder"], order)
                self.assertEqual(len(axes), count + 1)
                self.assertEqual(fig.get_figwidth(), baseline.get_figwidth())
                self.assertGreater(fig.get_figheight(), previousHeight)
                previousHeight = fig.get_figheight()
                np.testing.assert_allclose(axes[0].get_xlim(), baselineAxes[0].get_xlim())
                np.testing.assert_allclose(axes[0].get_ylim(), baselineAxes[0].get_ylim())
                layout = axes[0].get_position()
                np.testing.assert_allclose(
                    layout.width * fig.get_figwidth() / np.diff(axes[0].get_xlim()),
                    layout.height * fig.get_figheight() / np.diff(axes[0].get_ylim()))
                for axis, track in zip(axes[1:], tracks):
                    self.assertEqual(axis.get_ylabel(), f"{track.name} (mm)")
                    np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                                               [row[track.name] for row in coefficients], atol=1e-12)
                    np.testing.assert_allclose([bar.get_x() for bar in axis.patches],
                                               [0., 5., 13., 15., 19.])
                    np.testing.assert_allclose([bar.get_width() for bar in axis.patches],
                                               [5., 8., 2., 4., 15.])
                    bounds = axis.get_position()
                    self.assertAlmostEqual(bounds.x0, layout.x0)
                    self.assertAlmostEqual(bounds.width, layout.width)
                    self.assertAlmostEqual(bounds.height * fig.get_figheight(),
                                           baselineTrackHeight, delta=0.3 * baselineTrackHeight)
                plt.close(fig)

    def test_interleaved_seidel_and_wavefront_tracks_keep_order_and_calculations(self):
        lens = objective()
        tracks = [SD.SurfaceDataType.W060, SD.SurfaceDataType.SeidelW131,
                  SD.SurfaceDataType.W040, SD.SurfaceDataType.SeidelW040,
                  SD.SurfaceDataType.W242, SD.SurfaceDataType.W080]
        wavefront = lens.ComputeHigherOrder(referenceField=6., referenceFieldUnits="degrees")
        seidel = lens.ComputeSeidelCoefficients(referenceField=6.)
        wavefrontRows = [row.AsDict() for row in wavefront.CumulativeWavefrontCoefficients()]
        seidelRows = [row.AsDict() for row in seidel.CumulativeCoefficients()]
        with patch.object(lens, "ComputeHigherOrder", wraps=lens.ComputeHigherOrder) as compute, \
             patch.object(lens, "ComputeSeidelCoefficients", wraps=lens.ComputeSeidelCoefficients) as classical:
            fig, axes = lens.PlotSurfaceData(
                DisplayConfig=[SD.SurfaceDataType.RefractiveIndex, *tracks],
                SeidelReferenceField=6., PlotTrackLength=100.)
            self.assertEqual(compute.call_count, 1)
            self.assertEqual(classical.call_count, 1)
        self.assertEqual(axes[1].get_ylabel(), "nd")
        self.assertEqual([axis.get_ylabel() for axis in axes[2:]],
                         ["W060 (mm)", "Σ W131 (mm)", "W040 (mm)",
                          "Σ W040 (mm)", "W242 (mm)", "W080 (mm)"])
        for axis, track in zip(axes[2:], tracks):
            rows = seidelRows if track.name.startswith("Seidel") else wavefrontRows
            key = track.name.removeprefix("Seidel")
            np.testing.assert_allclose([bar.get_height() for bar in axis.patches],
                                       [row[key] for row in rows])

    def test_ten_tracks_stay_aligned_after_resize_and_export(self):
        lens = objective()
        # Module-level selection and explicit DisplayConfig use the same path.
        with patch.object(SD, "displayConfig", SD.displayConfigAberrations):
            fig, axes = lens.PlotSurfaceData(PlotTrackLength=100.)
        self.assertEqual(len(axes), 11)
        vertices = np.array([float(surface.cumulativeThickness) for surface in lens.surfaces])
        points = np.column_stack([vertices, np.zeros(len(vertices))])
        for width in [12., 20.]:
            fig.set_size_inches(width, fig.get_figheight())
            fig.canvas.draw()
            expected = axes[0].transData.transform(points)[:, 0]
            for axis in axes[1:]:
                np.testing.assert_allclose(axis.transData.transform(points)[:, 0],
                                           expected, rtol=0., atol=1e-7)
        fig.savefig(BytesIO(), format="png", dpi=80)
        expected = axes[0].transData.transform(points)[:, 0]
        for axis in axes[1:]:
            np.testing.assert_allclose(axis.transData.transform(points)[:, 0],
                                       expected, rtol=0., atol=1e-7)

    def test_chromatic_preset_uses_explicit_names_and_seidel_calculation(self):
        lens = objective()
        self.assertIn(SD.SurfaceDataType.SeidelW131C, SD.displayConfigSeidelC)
        self.assertEqual(len(SD.SurfaceDataType.__members__), len(SD.SurfaceDataType))
        with patch.object(lens, "ComputeHigherOrder", wraps=lens.ComputeHigherOrder) as compute:
            fig, axes = lens.PlotSurfaceData(DisplayConfig=SD.displayConfigSeidelC,
                                             PlotTrackLength=100.)
            compute.assert_not_called()
        self.assertEqual(len(axes), len(SD.chromaticLines) + 3)
        self.assertIn("Σ W131", axes[1].get_ylabel())

    def test_export_retains_full_width_and_height_after_screen_resize(self):
        lens = objective()
        fig, axes = lens.PlotSurfaceData(DisplayConfig=SD.displayConfigAberrations,
                                         PlotTrackLength=100.)
        exportSize = fig.get_size_inches().copy()
        # A desktop window can fit only the first ten inches of a tall canvas.
        fig.set_size_inches(16., 10.)
        fig.canvas.draw()
        output = BytesIO()
        exportWidths = []
        labelLeftEdges = []

        def recordExport(event):
            if event.canvas.is_saving():
                exportWidths.append(axes[0].get_position().width)
                labelLeftEdges.extend(axis.yaxis.label.get_window_extent(event.renderer).x0
                                      for axis in axes)

        fig.canvas.mpl_connect("draw_event", recordExport)
        fig.savefig(output, format="png", dpi=80)
        output.seek(0)
        with Image.open(output) as saved:
            self.assertEqual(saved.size, tuple((exportSize * 80).astype(int)))
        self.assertGreater(exportWidths[-1], 0.9)
        self.assertGreaterEqual(min(labelLeftEdges), 0.)
        np.testing.assert_allclose(fig.get_size_inches(), [16., 10.])
        np.testing.assert_allclose(fig.exportSize, exportSize)

    def test_explicit_data_track_height_controls_rows_and_export(self):
        for count in (5, 10):
            previousHeight = 0.
            for height in (0.65, 1.5):
                with self.subTest(count=count, height=height):
                    fig, axes = objective().PlotSurfaceData(
                        DisplayConfig=SD.displayConfigAberrations[:count + 3],
                        PlotTrackLength=100., DataTrackHeight=height)
                    self.assertEqual(fig.get_figwidth(), 16.)
                    self.assertGreater(fig.get_figheight(), previousHeight)
                    previousHeight = fig.get_figheight()
                    for axis in axes[1:]:
                        self.assertAlmostEqual(axis.get_position().height * fig.get_figheight(),
                                               height, delta=0.02)
                    fig.set_size_inches(12., 8.)
                    fig.canvas.draw()
                    savedHeights = []

                    def recordExport(event):
                        if event.canvas.is_saving():
                            savedHeights.extend(axis.get_position().height * fig.get_figheight()
                                                for axis in axes[1:])

                    fig.canvas.mpl_connect("draw_event", recordExport)
                    fig.savefig(BytesIO(), format="png", dpi=80)
                    np.testing.assert_allclose(savedHeights, height, atol=0.02)
                    plt.close(fig)


if __name__ == "__main__":
    unittest.main()
