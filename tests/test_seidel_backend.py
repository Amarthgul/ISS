"""Exercise actual material models and Seidel calculations on CPU and CUDA.

Each backend runs in a fresh process: existing framework modules bind their
backend at import time. This also prevents GPU tests changing other tests'
backend or importing NumPy-bound modules into a CUDA test.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
import json
import math
import sys
sys.path.insert(0, "src")
import Util.Backend as Backend
Backend.set_backend(sys.argv[1])
bd = Backend.backend
from Material import Material, MonochromaticMaterial
from Lens import Lens
from Surfaces.Stop import Stop
from Surfaces.Surface import Surface
from Surfaces.EvenAspheric import EvenAspheric
from Util.Analysis.Paraxial import SurfaceMatrix, SystemMatrix, TraceLens
from Util.Analysis.PAEFL import SingleGroup

def host(value):
    # Scalar-only conversion supports both NumPy and CuPy without implicit
    # NumPy conversion of a device array.
    if value.ndim == 0:
        return float(value)
    return [float(v) for v in value]

inputs = (587.56, 550, bd.asarray(587.56),
          bd.asarray([486, 588, 656]), [486, 588, 656])
materials = [Material("AIR"), Material("BAF9"), Material("BK7"),
             MonochromaticMaterial(1.37, "AIR")]
materialResults = []
for material in materials:
    samples = []
    for wavelength in inputs:
        result = material.RI(wavelength)
        assert result.dtype == bd.float64
        samples.append(host(result))
    materialResults.append(samples)

# Cover every implemented dispersion path with physical, nonsingular test
# coefficients. No catalog parsing or formula conventions are modified here.
formulaInputs = {
    "Schott": [2.25, .01, .001, .0001, .00001, .000001],
    "Conrady": [1.5, .01, .001],
    "Herzberger": [1.5, .001, .0001, .01, .001, .0001],
    "Sellmeier1": [.5, .01, .3, .02, .2, 100.0],
    "Sellmeier3": [.5, .01, .3, .02, .2, 100.0, .1, 200.0],
    "Sellmeier4": [1.5, .5, .01, .2, 100.0],
    "Sellmeier5": [.5, .01, .3, .02, .2, 100.0, .1, 200.0, .05, 300.0],
    "Extended 2": [2.25, .01, .001, .0001, .00001, .000001, .0000001, .00000001],
    "Extended 3": [2.25, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .001, .0001],
}
formulaResults = []
for formula, coefficients in formulaInputs.items():
    material = Material("AIR")
    material.name = "synthetic"
    material.Formula = formula
    material.coef = bd.asarray(coefficients, dtype=bd.float64)
    formulaResults.append([host(material.RI(wavelength)) for wavelength in inputs])

analyses = []
for environment in ("AIR", "BAF9"):
    for aspheric in (False, True):
        lens = Lens()
        lens.env = Material(environment)
        front = (EvenAspheric(50, 5, 20, "BAF9", -.3, [.001, 2e-6])
                 if aspheric else Surface(50, 5, 20, "BAF9"))
        lens.surfaces = [front, Surface(-50, 8, 20), Stop(2),
                         Surface(70, 4, 20, "BK7"), Surface(-60, 0, 20)]
        lens.stopIndex = 2
        lens.entrancePupil.clearSemiDiameter = bd.asarray(3.0)
        for settings in ({}, {"wavelength": 550},
                         {"fieldAngle": bd.asarray(6.0), "wavelength": bd.asarray(587.56),
                          "objectDistance": bd.asarray(math.inf)},
                         {"objectDistance": bd.asarray(800.0), "objectHeight": bd.asarray(35.0),
                          "pupilSemiDiameter": bd.asarray(2.0)},
                         {"stopSemiDiameter": bd.asarray(2.0)}):
            result = lens.ComputeSeidel(**settings)
            assert all(type(value) is float for value in result.totals.AsTuple())
            histories = []
            for row in result.paraxial.surfaces:
                histories.append([row.incidentRI, row.outgoingRI, row.curvature, row.quarticSag,
                                  row.marginalIncident.height, row.marginalIncident.slope,
                                  row.chiefIncident.height, row.chiefIncident.slope])
            analyses.append({"totals": result.totals.AsTuple(), "histories": histories,
                             "matrix": result.paraxial.systemMatrix,
                             "partial": result.SumSurfaces(1, 3).AsTuple()})
        # Direct entry points must also accept scalar wavelengths and GPU
        # indices without leaving device scalars in their host matrices.
        matrix, index = SurfaceMatrix(front, bd.asarray(1.0), 550)
        assert type(index) is float
        assert all(type(value) is float for row in matrix for value in row)
        matrix, index = SurfaceMatrix(lens.surfaces[2], bd.asarray(1.5), 550)
        assert type(index) is float
        analyses.append({"efl": float(SingleGroup(lens.surfaces[:2])),
                         "matrix": SystemMatrix(lens.surfaces, 550),
                         "geometry": front.ParaxialGeometry()})
print(json.dumps({"materials": materialResults, "formulas": formulaResults,
                  "analyses": analyses}, allow_nan=False))
'''


def probe(backend):
    result = subprocess.run([sys.executable, "-c", PROBE, backend], cwd=ROOT,
                            capture_output=True, text=True, timeout=60, check=True)
    return json.loads(result.stdout)


class BackendTests(unittest.TestCase):
    def test_cpu_material_scalar_array_contract(self):
        result = probe("CPU")
        self.assertEqual(result["materials"][0], [1., 1., 1., [1., 1., 1.], [1., 1., 1.]])
        self.assertEqual(len(result["formulas"]), 9)
        self.assertEqual(len(result["analyses"]), 24)

    @unittest.skipUnless(importlib.util.find_spec("cupy"), "CuPy is not installed")
    def test_cuda_matches_cpu(self):
        cpu, cuda = probe("CPU"), probe("CUDA")
        for key in ("materials", "formulas"):
            for cpuRows, cudaRows in zip(cpu[key], cuda[key]):
                for cpuValue, cudaValue in zip(cpuRows, cudaRows):
                    np.testing.assert_allclose(cudaValue, cpuValue, rtol=1e-12, atol=1e-14)
        for cpuRow, cudaRow in zip(cpu["analyses"], cuda["analyses"]):
            for key in cpuRow:
                np.testing.assert_allclose(cudaRow[key], cpuRow[key], rtol=1e-11, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
