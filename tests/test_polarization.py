"""Physics regressions for real coherency transport and its integration paths."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import Util.Backend as Backend
Backend.set_backend("CPU")

from Raytracing.Polarization import (
    TransverseBasis, DielectricCoherency, FresnelAmplitudes,
    InterfaceCoherency, TransportCoherency, TransformCoherency,
)
from Raytracing.RayBatch import RayBatch, GenerateBeam
from Raytracing.Emission import EmitField, EmitFromStop
from Raytracing.Refraction import Refract
from Raytracing.Reflection import Reflect
from Surfaces.Surface import Surface
from Surfaces.ClearBoundaryFlat import ClearBoundaryFlat
from Surfaces.MLA import MLA
from Imagers.Standard import StdImager
from Imagers.PDA import PDA
from ObjectSpace.Fog import FogAttenuator


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def rays(directions, terms=None):
    directions = unit(directions)
    data = np.zeros((len(directions), 13))
    data[:, 2] = -1
    data[:, 3:6] = directions
    data[:, 6] = 550
    data[:, 7:9] = 0.5
    data[:, 12] = np.arange(len(data)) + 10
    rb = RayBatch(data)
    if terms is not None:
        rb.SetRadianceTerms(np.asarray(terms, dtype=float))
    return rb


def medium(n):
    return SimpleNamespace(name="test dielectric", RI=lambda wavelengths: np.full(wavelengths.shape, n))


def plane(n=1.5):
    surface = Surface(np.inf, 1, 100, "BAF9")
    surface.material = medium(n)
    surface.SetCumulative(0.0)
    return surface


class CoherencyTests(unittest.TestCase):
    def test_basis_is_transverse_and_right_handed_including_axes(self):
        k = unit([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [1, 2, 3]])
        x, y = TransverseBasis(k)
        np.testing.assert_allclose(np.sum(x*k, axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.sum(y*k, axis=1), 0, atol=1e-15)
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1)
        np.testing.assert_allclose(np.cross(x, y), k, atol=1e-15)

    def test_analyzer_obeys_malus_law_and_orthogonal_sum(self):
        angles = np.linspace(0, np.pi, 51)
        c, s = np.cos(angles), np.sin(angles)
        terms = np.tile([1., 0., 0.], (len(angles), 1))
        rotated = TransformCoherency(terms, c, s, -s, c)
        np.testing.assert_allclose(rotated[:, 0], c*c, atol=1e-15)
        np.testing.assert_allclose(rotated[:, 0]+rotated[:, 2], 1)

    def test_normal_incidence_has_four_percent_reflection(self):
        for direction in ([0, 0, 1], [1, 0, 0], [0, -1, 0]):
            k = np.array([direction], dtype=float)
            C = np.array([[.7, .2, .3]])
            transmitted = DielectricCoherency(C, k, -k, k, 1., 1.5)
            reflected = DielectricCoherency(C, k, -k, -k, 1., 1.5, True)
            np.testing.assert_allclose(transmitted, .96*C, atol=1e-15)
            self.assertAlmostEqual((reflected[:, 0]+reflected[:, 2])[0], .04)

    def test_brewster_extinction_is_zero_not_infinite(self):
        angle = np.arctan(1.5)
        k = np.array([[np.sin(angle), 0, np.cos(angle)]])
        n = np.array([[0., 0., -1.]])
        # Canonical x is y-world (s), canonical y is p.
        C = np.array([[0., 0., 1.]])
        reflected = DielectricCoherency(C, k, n, Reflect(k, n), 1., 1.5, True)
        np.testing.assert_allclose(reflected, 0, atol=1e-28)
        rb = rays(k, reflected)
        self.assertLess(rb.PolarizedRadiance()[0], 1e-28)

    def test_random_states_conserve_energy_and_remain_psd(self):
        rng = np.random.default_rng(813)
        k = unit(np.column_stack((rng.uniform(-.8, .8, (300, 2)), np.ones(300))))
        n = np.tile([0., 0., -1.], (len(k), 1))
        factor = rng.normal(size=(len(k), 2, 2))
        matrix = factor @ factor.transpose(0, 2, 1)
        C = matrix[:, [0, 0, 1], [0, 1, 1]]
        ni, nt = np.ones(len(k)), np.full(len(k), 1.5)
        kt, _, _ = Refract(k, n, ni, nt)
        cr = DielectricCoherency(C, k, n, Reflect(k, n), ni, nt, True)
        ct = DielectricCoherency(C, k, n, kt, ni, nt)
        np.testing.assert_allclose(cr[:, 0]+cr[:, 2]+ct[:, 0]+ct[:, 2], np.trace(matrix, axis1=1, axis2=2), rtol=1e-13)
        for branch in (cr, ct):
            self.assertTrue(np.all(branch[:, 0] >= -1e-14))
            self.assertTrue(np.all(branch[:, 0]*branch[:, 2]-branch[:, 1]**2 >= -1e-13))

    def test_successive_non_coplanar_reflections_match_field_ensemble(self):
        k = unit([[.2, .1, 1.]])
        x, y = TransverseBasis(k)
        fields = np.array([1.2*x[0]+.4*y[0], .3*x[0]-.7*y[0]])
        local = fields @ np.stack((x[0], y[0]), axis=1)
        matrix = local.T @ local
        C = np.array([[matrix[0, 0], matrix[0, 1], matrix[1, 1]]])
        for normal in ([0, 0, -1], [.3, .6, 1], [-.5, .2, -1]):
            n = unit([normal])
            ko = Reflect(k, n)
            ci = abs(np.dot(k[0], n[0]))
            ct = np.sqrt(1-(1/1.5)**2*(1-ci*ci))
            rs = (ci-1.5*ct)/(ci+1.5*ct)
            rp = (1.5*ci-ct)/(1.5*ci+ct)
            s = unit(np.cross(k[0], n[0]))
            pi, po = np.cross(k[0], s), np.cross(ko[0], s)
            fields = rs*(fields@s)[:, None]*s + rp*(fields@pi)[:, None]*po
            C = DielectricCoherency(C, k, n, ko, 1., 1.5, True)
            xo, yo = TransverseBasis(ko)
            local = fields @ np.stack((xo[0], yo[0]), axis=1)
            expected = local.T @ local
            np.testing.assert_allclose(C[0], expected[[0, 0, 1], [0, 1, 1]], rtol=1e-12, atol=1e-16)
            k = ko

    def test_tir_is_energy_preserving_phase_free_approximation(self):
        k = unit([[1, 0, .2]])
        n = np.array([[0., 0., -1.]])
        rs, rp, ts, tp = FresnelAmplitudes(k, n, 1.5, 1.)
        np.testing.assert_array_equal([rs[0], rp[0], ts[0], tp[0]], [1, 1, 0, 0])
        C = np.array([[.6, .3, .4]])
        cr = DielectricCoherency(C, k, n, Reflect(k, n), 1.5, 1., True)
        self.assertAlmostEqual(cr[0, 0]+cr[0, 2], 1.)
        self.assertAlmostEqual(cr[0, 0]*cr[0, 2]-cr[0, 1]**2, .15)

    def test_equal_index_grazing_and_empty_batches(self):
        k = np.array([[1., 0., 0.]])
        n = np.array([[0., 0., 1.]])
        C = np.array([[.7, .1, .3]])
        np.testing.assert_allclose(DielectricCoherency(C, k, n, k, 1., 1.), C)
        empty = np.empty((0, 3))
        self.assertEqual(DielectricCoherency(empty, empty, empty, empty, np.empty(0), np.empty(0)).shape, (0, 3))
        self.assertEqual(TransportCoherency(empty, empty, empty).shape, (0, 3))

    def test_direction_transport_roundtrip_and_antiparallel(self):
        k = unit([[0, 0, 1], [.2, .3, 1], [1, 0, 0]])
        ko = unit([[1, 1, .1], [-.2, -.3, -1], [0, 0, 1]])
        C = np.tile([.7, .2, .3], (3, 1))
        moved = TransportCoherency(C, k, ko)
        np.testing.assert_allclose(moved[:, 0]+moved[:, 2], 1)
        np.testing.assert_allclose(TransportCoherency(moved, ko, k), C, atol=1e-14)

    def test_rigid_transform_rotates_polarization_about_ray(self):
        rb = rays([[0, 0, 1]], [[1, 0, 0]])
        transform = np.array([[0., -1, 0, 2], [1, 0, 0, 3], [0, 0, 1, 4], [0, 0, 0, 1]])
        rb.Transform(transform)
        np.testing.assert_allclose(rb.Direction(), [[0, 0, 1]])
        np.testing.assert_allclose(rb.RadianceTerms(), [[0, 0, 1]])
        np.testing.assert_allclose(rb.Position(), [[2, 3, 3]])

    def test_scaling_pruning_and_valid_singular_states(self):
        rb = rays([[0, 0, 1]]*3, [[1, 0, 0], [0, 0, 0], [.5, .5, .5]])
        rb.SanitizePolarization()
        self.assertEqual(len(rb.value), 3)
        rb.RadianceChange([.25, 0, .5])
        np.testing.assert_allclose(rb.PolarizedRadiance(False), [.25, 0, .5])
        rb.RadianceChange(0.)
        np.testing.assert_array_equal(rb.RadianceTerms(), 0)
        self.assertEqual(len(rb.RadiantKill().value), 0)


class PolarizationIntegrationTests(unittest.TestCase):
    def test_surface_transmission_independent_of_stray_generation(self):
        surface = plane()
        incoming = rays([[0, 0, 1]], [[.7, .2, .3]])
        original = incoming.value.copy()
        outputs = []
        for reflection in (False, True):
            main, tir, vig, stray = surface.Trace(incoming, np.ones(1), reflection=reflection)
            np.testing.assert_allclose(main.RadianceTerms(), [[.672, .192, .288]])
            np.testing.assert_array_equal(main.GetAoV(), incoming.GetAoV())
            outputs.append(main.value)
            if reflection:
                np.testing.assert_allclose(main.Radiance()+stray.Radiance(), 1.)
            else:
                self.assertIsNone(stray)
        np.testing.assert_allclose(outputs[0], outputs[1])
        np.testing.assert_array_equal(incoming.value, original)
        naive, _, _ = surface.NaiveTrace(incoming, np.ones(1))
        np.testing.assert_allclose(naive.RadianceTerms(), outputs[0][:, [7, 9, 8]])

    def test_surface_mixed_tir_and_transmission_preserve_payload(self):
        incoming = rays([[0, 0, 1], [1, 0, .2]])
        main, tir, _, stray = plane(1.).Trace(incoming, np.full(2, 1.5), reflection=True)
        np.testing.assert_array_equal(tir, [False, True])
        self.assertAlmostEqual(main.Radiance().sum()+stray.Radiance().sum(), 2.)
        np.testing.assert_array_equal(main.GetAoV(), [[10]])
        np.testing.assert_array_equal(stray.GetAoV(), [[10], [11]])

    def test_surface_reverse_and_all_vignetted(self):
        incoming = rays([[.2, .1, -1]])
        incoming.value[:, 2] = 1
        main, _, _, stray = plane().Trace(incoming, np.ones(1), inverted=True, reflection=True)
        self.assertAlmostEqual(main.Radiance().sum()+stray.Radiance().sum(), 1.)
        incoming.value[:, 0] = 1000
        main, _, _, stray = plane().Trace(incoming, np.ones(1), inverted=True, reflection=True)
        self.assertEqual(len(main.value), 0)
        self.assertEqual(len(stray.value), 0)

    def test_ideal_mirror_preserves_power_and_polarization_degree(self):
        surface = plane()
        rb = rays([[.2, .3, 1]], [[.6, .3, .4]])
        result, _, _, _ = surface.TraceMirror(rb, np.ones(1))
        self.assertAlmostEqual(result.Radiance()[0], 1.)
        self.assertAlmostEqual(np.linalg.det(result.PolarizationMat())[0], .15)

    def test_clear_boundary_specular_and_tir(self):
        boundary = ClearBoundaryFlat(np.array([[-10., -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]]))
        boundary.specularReflection = 1.
        boundary.absorption = 0.
        boundary.exteriorCoating = medium(1.5)
        result, mask = boundary.Trace(rays([[0, 0, 1]]), np.ones(1))
        np.testing.assert_allclose(result.Radiance(), [.04])
        boundary.exteriorCoating = medium(1.)
        result, mask = boundary.Trace(rays([[1, 0, .2]]), np.full(1, 1.5))
        np.testing.assert_allclose(result.Radiance(), [1.])

    def test_sensor_microlens_includes_fresnel_loss(self):
        sensor = PDA(w=2, h=2, horiPx=2)
        sensor.materialMLA = medium(1.5)
        result = sensor._ApplyMLARefraction(rays([[0, 0, 1]]), np.array([[0., 0., 1.]]))
        np.testing.assert_allclose(result.Radiance(), [.96])

    def test_mla_builds_valid_raybatch_and_transmits_power(self):
        mla = MLA(1., .002, "BAF9")
        mla.material = medium(1.5)
        mla.SetShape(1, 1, .006)
        incoming = rays([[0, 0, -1]])
        incoming.SetPosition(np.array([[.003, .003, 1.]]))
        result, tir, vig, _ = mla.Trace(incoming, np.ones(1))
        np.testing.assert_allclose(result.Radiance(), [.96])
        np.testing.assert_allclose(result.Direction(), [[0, 0, -1]])
        np.testing.assert_array_equal(result.GetAoV(), incoming.GetAoV())

    def test_emission_starts_at_unit_total_power(self):
        beam = GenerateBeam(np.zeros(3), np.array([0., 0., 1.]), size=4)
        field = EmitField(0., 0., sampleTargets=np.array([[0., 0., 0.], [.1, .1, 0.]]))
        for rb in (beam, field):
            np.testing.assert_allclose(rb.Radiance(), 1.)
            np.testing.assert_allclose(rb.value[:, 9], 0.)

    def test_stop_emission_preserves_launch_angles_as_aovs(self):
        beams = EmitFromStop(2, np.zeros(3), 1., 1., -2., 2., numRays=5)
        for rb in beams:
            np.testing.assert_allclose(rb.Radiance(), 1.)
            np.testing.assert_array_equal(rb.value[:, 9], 0.)
            np.testing.assert_array_equal(rb.SurfaceIndex(), 2.)
            np.testing.assert_allclose(rb.GetAoV()[:, 0], np.linspace(-np.arctan(.5), np.arctan(.5), 5))

    def test_fog_mixes_unpolarized_ambient_power_linearly(self):
        fog = FogAttenuator(ms=0, extinction_length=1, ambient_level=2, maxWhiten=0)
        fog.minDistance = 0
        rb = rays([[0, 0, 1]], [[.6, .2, .4]])
        rb.SetPosition(np.array([[0., 0., np.log(2.)]]))
        result = fog.Attenuate(rb)
        np.testing.assert_allclose(result.RadianceTerms(), [[.8, .1, .7]])
        np.testing.assert_allclose(rb.RadianceTerms(), [[.6, .2, .4]])

    def test_haze_transport_and_attenuation_preserve_degree(self):
        surface = plane()
        surface.hazeSigma = .1
        surface.hazeTransmissionLoss = .2
        rb = rays([[0, 0, 1]]*30, [[.6, .2, .4]]*30)
        result = surface._HazePass(rb, wavelengthScale=False)
        power = result.Radiance()
        self.assertTrue(np.all(power <= 1.+1e-14))
        self.assertTrue(np.all(power >= .8-1e-14))
        np.testing.assert_allclose(np.linalg.det(result.PolarizationMat()) / power**2, .2, atol=1e-14)

    def test_detector_sums_power_without_polarization_bias(self):
        detector = StdImager(w=2, h=2, horiPx=2)
        rb = rays([[0, 0, 1]]*3, [[1, 0, 0], [0, 0, 1], [.5, .5, .5]])
        rb.value[:, 2] = detector._zPos
        rb.value[:, 11] = [0, 1, 2]
        rb.value = rb.value[:, :12]
        for polarized in (True, False):
            image = detector._integralRaysChannelBased(rb, overExpNoiseRemoval=None, polarized=polarized)
            np.testing.assert_allclose(image.sum(axis=(0, 1)), [1, 1, 1])
        stats = detector._IncidentStats(rb)
        self.assertIn("invalid_any=0", stats)


if __name__ == "__main__":
    unittest.main()
