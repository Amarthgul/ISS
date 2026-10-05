"""Numerical checks for paraxial tracing and monochromatic Seidel sums."""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Util.Backend as Backend
Backend.set_backend("CPU")

from Lens import Lens
from Surfaces.Surface import Surface
from Surfaces.Stop import Stop
from Surfaces.EvenAspheric import EvenAspheric
from Util.Analysis.Paraxial import SystemMatrix, TraceLens
from Util.Analysis.PAEFL import SingleGroup
from Util.Analysis.Seidel import ComputeSeidel, CalculateSeidel, ComputeSeidelCoefficients


def surface(radius, thickness, index=1.0):
    result = Surface(radius, thickness, 20.0)
    # A nondispersive material avoids catalog dependence of reference values.
    result.material.RI = lambda wavelength: np.ones_like(wavelength) * index
    return result


def objective():
    lens = Lens()
    lens.surfaces = [surface(50.0, 5.0, 1.5), surface(-50.0, 8.0),
                     Stop(2.0), surface(70.0, 4.0, 1.6), surface(-60.0, 0.0)]
    lens.stopIndex = 2
    lens.entrancePupil.clearSemiDiameter = 3.0
    return lens


def values(coefficients):
    return np.array([coefficients.S1, coefficients.S2, coefficients.S3,
                     coefficients.S4, coefficients.S5])


def exact_image_position(lens, x, y, angle, imageZ):
    """Independent sag-intersection/vector Snell trace, with no ray helpers."""
    position = np.array([x, y - math.tan(angle), -1.0])
    direction = np.array([0.0, math.sin(angle), math.cos(angle)])
    previousRI = 1.0
    vertex = 0.0
    for s in lens.surfaces:
        if not s.stopOnly:
            c = 1.0 / float(s.radius)
            k, coefficients = (float(s.K), s.asphCoef) if s.cType.name == "EvenAspheric" else (0.0, [])

            def sag_and_derivative(h):
                root = math.sqrt(1.0 - (1.0 + k) * c*c * h*h)
                sag = c * h*h / (1.0 + root)
                derivative = c * h / root
                for i, coefficient in enumerate(coefficients):
                    power = 2 * (i + 1)
                    sag += float(coefficient) * h**power
                    derivative += power * float(coefficient) * h**(power - 1)
                return sag, derivative

            distance = (vertex - position[2]) / direction[2]
            for _ in range(12):
                hit = position + distance * direction
                radius = np.linalg.norm(hit[:2])
                sag, derivative = sag_and_derivative(radius)
                gradient = derivative * hit[:2] / radius if radius else np.zeros(2)
                distance -= (hit[2] - vertex - sag) / (direction[2] - gradient @ direction[:2])
            position += distance * direction
            normal = np.array([-gradient[0], -gradient[1], 1.0])
            normal /= np.linalg.norm(normal)
            currentRI = float(s.RI(550.0))
            eta = previousRI / currentRI
            cosine = direction @ normal
            direction = eta * direction + (math.sqrt(1 - eta**2 * (1 - cosine**2)) - eta * cosine) * normal
            previousRI = currentRI
        vertex += float(s.thickness)
    return position[:2] + (imageZ - position[2]) * direction[:2] / direction[2]


class SeidelTests(unittest.TestCase):
    def test_lens_default_reference_preserves_positive_angular_field(self):
        # PlotSurfaceData forwards the reference field and wavelength only.
        # Its positive angular field must not fall through to object height.
        lens = objective()
        reference = lens.ComputeSeidelCoefficients(referenceField=20.0,
                                                  wavelength=587.56)
        trace = reference.seidel.paraxial
        self.assertEqual(reference.fieldUnits, "degrees")
        self.assertEqual(trace.fieldAngle, 20.0)
        self.assertAlmostEqual(trace.surfaces[0].chiefIncident.slope,
                               math.radians(20.0))
        self.assertAlmostEqual(trace.surfaces[lens.stopIndex].chiefIncident.height, 0.0)

    def test_defocused_conjugates_preserve_five_terms_and_prescription(self):
        lens = objective()
        thicknesses = [float(s.thickness) for s in lens.surfaces]
        settings = dict(objectDistance=13500.0, referenceField=10.0,
                        referenceFieldUnits="degrees", stopSemiDiameter=2.0)
        matched = lens.ComputeSeidelCoefficients(focusDistance=None, **settings)
        explicit = lens.ComputeSeidelCoefficients(focusDistance=13500.0, **settings)
        defocused = lens.ComputeSeidelCoefficients(focusDistance=1350.0, **settings)
        for result in (explicit, defocused):
            np.testing.assert_array_equal(result.totals.AsTuple(), matched.totals.AsTuple())
            for actual, expected in zip(result.surfaces, matched.surfaces):
                self.assertEqual(actual, expected)
            self.assertEqual(result.CumulativeCoefficients(), matched.CumulativeCoefficients())
            self.assertEqual(result.SumSurfaces(1, 3), matched.SumSurfaces(1, 3))
            self.assertEqual(len(result.totals.AsTuple()), 5)
        self.assertEqual(matched.seidel.paraxial.focusDistance, 13500.0)
        self.assertEqual(matched.seidel.paraxial.imageDefocus, 0.0)
        self.assertGreater(defocused.seidel.paraxial.imageDefocus, 0.0)
        focusedObject = lens.ComputeSeidelCoefficients(
            objectDistance=1350.0, focusDistance=1350.0, referenceField=10.0,
            referenceFieldUnits="degrees", stopSemiDiameter=2.0)
        self.assertFalse(np.allclose(defocused.totals.AsTuple(),
                                     focusedObject.totals.AsTuple(), rtol=1e-5, atol=0.0))
        self.assertEqual([float(s.thickness) for s in lens.surfaces], thicknesses)
        self.assertIsNone(lens.focalPoint)
        self.assertIsNone(lens.surfaces[0].frontVertex)

    def test_focus_reference_matches_independent_interface_conjugates(self):
        lens = objective()
        lens.surfaces = [Stop(0.0), surface(50.0, 0.0, 1.5)]
        lens.stopIndex = 0
        for actual, focus in ((13500.0, 1350.0), (math.inf, 1350.0),
                              (13500.0, math.inf), (math.inf, math.inf)):
            trace = TraceLens(lens, objectDistance=actual, focusDistance=focus)
            # Independent Gaussian refraction: n'/L = (n'-n)/R - n/d.
            actualImage = 1.5 / (.5 / 50.0 - 1.0 / actual)
            focusImage = 1.5 / (.5 / 50.0 - 1.0 / focus)
            self.assertAlmostEqual(trace.objectImageZ, actualImage)
            self.assertAlmostEqual(trace.focusImageZ, focusImage)
            self.assertAlmostEqual(trace.imageDefocus, focusImage - actualImage)
        # A plane interface at infinite conjugates remains usable by Seidel.
        lens.surfaces[1] = surface(math.inf, 0.0, 1.5)
        trace = TraceLens(lens)
        self.assertTrue(math.isinf(trace.objectImageZ))
        self.assertTrue(math.isinf(trace.focusImageZ))
        self.assertEqual(trace.imageDefocus, 0.0)

    def test_finite_angular_field_aiming_and_evaluation(self):
        lens = objective()
        reference = lens.ComputeSeidelCoefficients(
            referenceField=7.0, referenceFieldUnits="degrees",
            objectDistance=13500.0, focusDistance=1350.0, stopSemiDiameter=2.0)
        trace = reference.seidel.paraxial
        self.assertEqual(reference.fieldUnits, "degrees")
        self.assertEqual(trace.fieldAngle, 7.0)
        self.assertAlmostEqual(trace.surfaces[0].chiefIncident.slope, math.radians(7.0))
        self.assertAlmostEqual(trace.surfaces[lens.stopIndex].chiefIncident.height, 0.0)
        initial = trace.surfaces[0].chiefIncident
        self.assertAlmostEqual(initial.height - trace.objectDistance * initial.slope,
                               trace.objectHeight)
        heightBased = lens.ComputeSeidel(
            objectDistance=13500.0, objectHeight=trace.objectHeight,
            focusDistance=1350.0, stopSemiDiameter=2.0)
        np.testing.assert_allclose(values(heightBased.totals), values(reference.seidel.totals),
                                   rtol=1e-13, atol=1e-15)
        for field in (-12.0, 0.0, 3.0, 7.0):
            evaluated = reference.EvaluateField(field)
            direct = lens.ComputeSeidel(
                fieldAngle=field, objectHeight=None, objectDistance=13500.0,
                focusDistance=1350.0, stopSemiDiameter=2.0)
            np.testing.assert_allclose(values(evaluated.totals), values(direct.totals), atol=1e-15)
            self.assertEqual(evaluated.paraxial.fieldAngle, field)
            self.assertAlmostEqual(evaluated.paraxial.objectHeight, direct.paraxial.objectHeight)
            self.assertEqual(evaluated.paraxial.focusDistance, 1350.0)
            self.assertEqual(evaluated.paraxial.imageDefocus, trace.imageDefocus)
            for actual, expected in zip(evaluated.paraxial.surfaces, direct.paraxial.surfaces):
                self.assertAlmostEqual(actual.chiefIncident.height, expected.chiefIncident.height)
                self.assertAlmostEqual(actual.chiefOutgoing.slope, expected.chiefOutgoing.slope)

    def test_finite_angular_field_has_infinite_conjugate_limit(self):
        lens = objective()
        settings = dict(referenceField=3.0, referenceFieldUnits="degrees",
                        focusDistance=1350.0, stopSemiDiameter=2.0)
        infinite = ComputeSeidelCoefficients(lens, **settings)
        for distance in (1350.0, 13500.0):
            finite = ComputeSeidelCoefficients(lens, objectDistance=distance, **settings)
            self.assertEqual(finite.seidel.paraxial.surfaces[0].chiefIncident,
                             infinite.seidel.paraxial.surfaces[0].chiefIncident)
        distant = ComputeSeidelCoefficients(lens, objectDistance=1e12, **settings)
        np.testing.assert_allclose(distant.totals.AsTuple(), infinite.totals.AsTuple(), rtol=1e-8)

    def test_defocused_finite_spherical_matches_exact_snell_trace(self):
        lens = objective()
        lens.surfaces = [Stop(0.0), surface(50.0, 5.0, 1.5), surface(-50.0, 0.0)]
        lens.stopIndex = 0
        result = ComputeSeidel(lens, objectDistance=13500.0, focusDistance=1350.0,
                               objectHeight=0.0, pupilSemiDiameter=1.0)
        last = result.paraxial.surfaces[-1]
        predicted = result.totals.S1 / (2.0 * last.outgoingRI * last.marginalOutgoing.slope)
        errors = []
        for height in (.2, .1):
            # Launch the finite-conjugate axial ray at the first vertex.
            angle = math.atan(height / 13500.0)
            errors.append(exact_image_position(lens, 0.0, height, angle,
                                               result.paraxial.objectImageZ)[1] / height**3)
        extrapolated = (4 * errors[1] - errors[0]) / 3
        self.assertAlmostEqual(extrapolated, predicted, delta=abs(predicted) * 1e-6)

    def test_reference_coefficients_survive_zero_field(self):
        lens = objective()
        reference = lens.ComputeSeidelCoefficients(referenceField=5.0)
        selected = reference.EvaluateField(0.0)
        np.testing.assert_array_equal(values(selected.totals)[1:], np.zeros(4))
        self.assertNotEqual(reference.totals.W131, 0.0)
        self.assertNotEqual(reference.totals.W222, 0.0)
        self.assertNotEqual(reference.totals.W311, 0.0)
        self.assertAlmostEqual(selected.totals.S1 / 8, reference.totals.W040)
        self.assertEqual(reference.seidel.paraxial.fieldAngle, 5.0)
        self.assertEqual(selected.paraxial.fieldAngle, 0.0)
        self.assertEqual(selected.paraxial.lagrangeInvariant, 0.0)

    def test_reference_evaluation_matches_direct_full_trace(self):
        lens = objective()
        for conjugate in ({}, {"objectDistance": 800.0}):
            reference = ComputeSeidelCoefficients(lens, referenceField=7.0, **conjugate)
            for field in (-12.0, 0.0, 3.0, 7.0):
                evaluated = reference.EvaluateField(field)
                if conjugate:
                    direct = ComputeSeidel(lens, objectHeight=field, **conjugate)
                else:
                    direct = ComputeSeidel(lens, fieldAngle=field)
                np.testing.assert_allclose(values(evaluated.totals), values(direct.totals), atol=1e-15)
                for actual, expected in zip(evaluated.surfaces, direct.surfaces):
                    np.testing.assert_allclose(values(actual.coefficients), values(expected.coefficients), atol=1e-15)
                for actual, expected in zip(evaluated.paraxial.surfaces, direct.paraxial.surfaces):
                    self.assertAlmostEqual(actual.chiefIncident.height, expected.chiefIncident.height)
                    self.assertAlmostEqual(actual.chiefOutgoing.slope, expected.chiefOutgoing.slope)
            self.assertEqual(reference.fieldUnits, "mm" if conjugate else "degrees")

    def test_cumulative_reference_coefficients_and_conversion(self):
        lens = objective()
        reference = ComputeSeidelCoefficients(lens, referenceField=5.0)
        increments = np.array([row.coefficients.AsTuple() for row in reference.surfaces])
        cumulative = np.array([row.AsTuple() for row in reference.CumulativeCoefficients()])
        np.testing.assert_allclose(cumulative, np.cumsum(increments, axis=0), atol=1e-15)
        np.testing.assert_allclose(cumulative[-1], reference.totals.AsTuple())
        np.testing.assert_allclose(reference.SumSurfaces(1, 3).AsTuple(), increments[1:4].sum(axis=0))
        np.testing.assert_allclose(values(reference.seidel.CumulativeSums()[-1]), values(reference.seidel.totals))
        for row, raw in zip(reference.surfaces, reference.seidel.surfaces):
            wave = row.coefficients
            np.testing.assert_allclose([8*wave.W040, 2*wave.W131, 2*wave.W222,
                                       4*wave.W220 - 2*wave.W222, 2*wave.W311],
                                       values(raw.coefficients), atol=1e-15)
            np.testing.assert_allclose(row.coefficients.AsTuple(),
                                       np.array(row.sphericalBase.AsTuple()) + row.asphericDeparture.AsTuple())

    def test_reference_field_defines_normalization_not_evaluation_field(self):
        lens = objective()
        first = ComputeSeidelCoefficients(lens, referenceField=2.0)
        second = ComputeSeidelCoefficients(lens, referenceField=4.0)
        np.testing.assert_allclose(second.totals.AsTuple(),
                                   np.array(first.totals.AsTuple()) * [1, 2, 4, 4, 8])
        np.testing.assert_allclose(values(first.EvaluateField(6).totals),
                                   values(second.EvaluateField(6).totals), atol=1e-15)

    def test_thick_lens_efl_and_stop_medium(self):
        surfaces = [surface(50.0, 5.0, 1.5), surface(-50.0, 0.0)]
        self.assertAlmostEqual(float(SingleGroup(surfaces)), 3000.0 / 59.0)
        split = [surface(50.0, 2.0, 1.5), Stop(3.0), surface(-50.0, 0.0)]
        np.testing.assert_allclose(SystemMatrix(split), SystemMatrix(surfaces), atol=1e-15)
        self.assertTrue(math.isinf(float(SingleGroup([]))))
        self.assertTrue(math.isinf(float(SingleGroup([surface(math.inf, 0)]))))

    def test_single_interface_reference_coefficients(self):
        lens = objective()
        lens.surfaces = [Stop(0.0), surface(50.0, 0.0, 1.5)]
        lens.stopIndex = 0
        result = lens.ComputeSeidel(fieldAngle=math.degrees(.01), pupilSemiDiameter=1.0)
        np.testing.assert_allclose(values(result.totals),
                                   np.array([16/9, 8/9, 4/9, 2/3, 5/9]) * 1e-6,
                                   rtol=1e-13)
        np.testing.assert_array_equal(values(result.surfaces[0].coefficients), np.zeros(5))

    def test_aiming_invariant_and_finite_conjugate(self):
        lens = objective()
        for settings in ({"fieldAngle": 6.0},
                         {"objectDistance": 800.0, "objectHeight": 35.0}):
            trace = TraceLens(lens, **settings)
            self.assertAlmostEqual(trace.surfaces[lens.stopIndex].chiefIncident.height, 0.0)
            for row in trace.surfaces:
                for n, marginal, chief in ((row.incidentRI, row.marginalIncident, row.chiefIncident),
                                           (row.outgoingRI, row.marginalOutgoing, row.chiefOutgoing)):
                    invariant = n * (marginal.height * chief.slope - chief.height * marginal.slope)
                    self.assertAlmostEqual(invariant, trace.lagrangeInvariant, places=14)
            self.assertEqual(trace.surfaces[0].vertexZ, 0.0)
            if math.isfinite(trace.objectDistance):
                first = trace.surfaces[0]
                self.assertAlmostEqual(first.marginalIncident.height - 800 * first.marginalIncident.slope, 0.0)
                self.assertAlmostEqual(first.chiefIncident.height - 800 * first.chiefIncident.slope, 35.0)
            explicit = TraceLens(lens, stopSemiDiameter=trace.stopSemiDiameter, **settings)
            np.testing.assert_allclose(values(CalculateSeidel(explicit).totals),
                                       values(CalculateSeidel(trace).totals), rtol=1e-13)

    def test_scaling_field_reversal_and_full_system_sums(self):
        lens = objective()
        result = ComputeSeidel(lens, fieldAngle=4.0)
        doubledPupil = ComputeSeidel(lens, fieldAngle=4.0, pupilSemiDiameter=6.0)
        doubledField = ComputeSeidel(lens, fieldAngle=8.0)
        reversedField = ComputeSeidel(lens, fieldAngle=-4.0)
        np.testing.assert_allclose(values(doubledPupil.totals), values(result.totals) * [16, 8, 4, 4, 2])
        np.testing.assert_allclose(values(doubledField.totals), values(result.totals) * [1, 2, 4, 4, 8])
        np.testing.assert_allclose(values(reversedField.totals), values(result.totals) * [1, -1, 1, 1, -1])
        np.testing.assert_allclose(values(result.SumSurfaces()), values(result.totals))
        np.testing.assert_allclose(values(result.SumSurfaces(1, 3)),
                                   sum((values(row.coefficients) for row in result.surfaces[1:4])))
        # The pure stop retains its incoming glass/air medium and contributes zero.
        np.testing.assert_array_equal(values(result.surfaces[2].coefficients), np.zeros(5))
        self.assertEqual(lens.focalPoint, None)
        self.assertEqual(lens.surfaces[0].frontVertex, None)

    def test_asphere_a2_and_equivalent_sphere(self):
        s = EvenAspheric(50.0, 0.0, 10.0, "AIR", -.3, [.001, 2e-6, .01])
        c, q = s.ParaxialGeometry()
        self.assertAlmostEqual(c, .022)
        self.assertAlmostEqual(q, .7 * .02**3 / 8 + 2e-6)
        lens = objective()
        lens.surfaces = [Stop(0.0), s]
        lens.stopIndex = 0
        s.material.RI = lambda wavelength: np.ones_like(wavelength) * 1.5
        self.assertAlmostEqual(float(SingleGroup([s])), 3.0 / .022)
        s.K = 0.0
        s.asphCoef = np.array([.001, (.022**3 - .02**3)/8])
        result = ComputeSeidel(lens)
        lens.surfaces[1] = surface(1/.022, 0.0, 1.5)
        np.testing.assert_allclose(values(result.totals), values(ComputeSeidel(lens).totals), atol=1e-15)

    def test_normal_marginal_incidence_is_regular(self):
        # At the second surface u = -c*y, hence A = 0. Division-form S5
        # would fail here although the physical contribution is finite.
        lens = objective()
        lens.surfaces = [Stop(0.0), surface(50.0, 0.0, 1.5), surface(150.0, 0.0)]
        lens.stopIndex = 0
        result = ComputeSeidel(lens, fieldAngle=5.0)
        self.assertTrue(np.isfinite(values(result.surfaces[-1].coefficients)).all())
        self.assertNotEqual(result.surfaces[-1].coefficients.S5, 0.0)

    def test_zero_marginal_height_at_asphere_is_regular(self):
        lens = objective()
        asphere = EvenAspheric(50.0, 0.0, 20.0, "AIR", -.5, [0.0, 1e-6])
        lens.surfaces = [Stop(0.0), surface(50.0, 150.0, 1.5), asphere]
        lens.stopIndex = 0
        result = ComputeSeidel(lens, fieldAngle=5.0)
        self.assertAlmostEqual(result.paraxial.surfaces[-1].marginalIncident.height, 0.0)
        self.assertTrue(np.isfinite(values(result.totals)).all())

    def test_primary_spherical_matches_independent_exact_snell_trace(self):
        for aspheric in (False, True):
            lens = objective()
            lens.surfaces = [Stop(0.0), surface(50.0, 5.0, 1.5), surface(-50.0, 0.0)]
            lens.stopIndex = 0
            if aspheric:
                s = EvenAspheric(50.0, 5.0, 20.0, "AIR", -.4, [.001, 8e-7])
                s.material.RI = lambda wavelength: np.ones_like(wavelength) * 1.5
                lens.surfaces[1] = s
            result = ComputeSeidel(lens, fieldAngle=0.0, pupilSemiDiameter=1.0)
            last = result.paraxial.surfaces[-1]
            ray = last.marginalOutgoing
            imageZ = last.vertexZ - ray.height / ray.slope
            predicted = result.totals.S1 / (2.0 * last.outgoingRI * ray.slope)
            # Two small apertures extrapolate away the fifth-order ray error.
            first = exact_image_position(lens, 0, .2, 0, imageZ)[1] / .2**3
            second = exact_image_position(lens, 0, .1, 0, imageZ)[1] / .1**3
            extrapolated = (4 * second - first) / 3
            self.assertAlmostEqual(extrapolated, predicted, delta=abs(predicted)*1e-6)

    def test_all_five_terms_match_exact_off_axis_ray_errors(self):
        lens = objective()
        lens.surfaces = [Stop(8.0), surface(50.0, 5.0, 1.5), surface(-50.0, 0.0)]
        lens.stopIndex = 0
        for aspheric in (False, True):
            if aspheric:
                s = EvenAspheric(50.0, 5.0, 20.0, "AIR", -.4, [.001, 8e-7])
                s.material.RI = lambda wavelength: np.ones_like(wavelength) * 1.5
                lens.surfaces[1] = s
            angle, pupil = .06, 3.0
            result = ComputeSeidel(lens, math.degrees(angle), pupilSemiDiameter=pupil)
            last = result.paraxial.surfaces[-1]
            imageZ = last.vertexZ - last.marginalOutgoing.height / last.marginalOutgoing.slope
            imageDistance = imageZ - last.vertexZ
            imageScale = (result.paraxial.systemMatrix[0][1] +
                          imageDistance * result.paraxial.systemMatrix[1][1])
            s1, s2, s3, s4, s5 = values(result.totals)
            denominator = 2 * last.outgoingRI * last.marginalOutgoing.slope
            for px, py in ((0, 0), (.7, 0), (0, .7), (.5, -.4)):
                radiusSquared = px*px + py*py
                expected = np.array([
                    s1*radiusSquared*px + 2*s2*px*py + (s3+s4)*px,
                    s1*radiusSquared*py + s2*(px*px+3*py*py) + (3*s3+s4)*py+s5]) / denominator
                errors = []
                for scale in (.2, .1):
                    actual = exact_image_position(lens, scale*pupil*px, scale*pupil*py,
                                                  scale*angle, imageZ)
                    actual[1] -= imageScale * math.tan(scale*angle)
                    errors.append(actual / scale**3)
                extrapolated = (4*errors[1] - errors[0]) / 3
                np.testing.assert_allclose(extrapolated, expected, rtol=3e-5, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
