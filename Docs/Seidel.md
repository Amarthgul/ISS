# Monochromatic Seidel analysis

`Util.Analysis.Paraxial` traces first-order marginal and chief rays through a
complete centered refractive prescription. `Util.Analysis.Seidel` uses those
histories to calculate the five monochromatic primary aberrations, without
real-ray sampling, image formation, or plotting.

```python
from Util.Globals import LambdaLines

# lens is a correctly defined Lens with stopIndex and current entrance pupil.
result = lens.ComputeSeidel(fieldAngle=10.0, wavelength=LambdaLines["d"])

print(result.totals.AsDict())
for surface in result.surfaces:
    print(surface.surfaceIndex, surface.coefficients.AsTuple())

# Inclusive, zero-based surface range, retaining the full-system ray trace.
print(result.SumSurfaces(first=2, last=5).AsDict())
```

The standalone function `Util.Analysis.Seidel.ComputeSeidel(lens, ...)` accepts
the same settings. `CalculateSeidel(paraxial)` accepts an existing result from
`Util.Analysis.Paraxial.TraceLens(lens, ...)`, so inspecting rays and calculating
coefficients need not trace the lens twice.

## Inputs and normalization

- Lengths are mm, wavelengths are numeric nm, and surface indices are zero-based.
- `fieldAngle` is the signed incident paraxial slope expressed in degrees for an
  infinite object. The default is a **1-degree reference field**, not the lens's
  maximum photographic field. Supply the field you want to evaluate.
- `objectDistance` defaults to infinity. For a finite object, supply its positive
  distance back from the first vertex and its signed `objectHeight` in mm.
  `fieldAngle` is ignored for finite objects. Heights and slopes refer to the
  positive meridional Y direction; propagation is toward positive Z.
- `pupilSemiDiameter` defaults to `lens.entrancePupil.clearSemiDiameter`, the
  current working pupil radius, not the maximum full-open diameter. At finite
  conjugates this radius is applied at the paraxial entrance-pupil plane derived
  from the stop transfer matrix. The sampled pupil's depth/profile is not used.
- `stopSemiDiameter`, when supplied, instead normalizes the marginal ray at the
  aperture stop. It takes precedence over `pupilSemiDiameter`. The result records
  the resulting pupil radius and signed marginal height at the stop.

The lens must be nonempty, centered, refractive, have a defined `stopIndex`, and
have nonsingular stop aiming and a finite pupil radius when no stop radius is
supplied. Spherical and even-aspheric surfaces are supported. Mirrors,
decenters, biconics, diffraction, pupil masks, clipping and chromatic aberrations
are outside this calculation's contract. Aperture stops preserve their incident
medium. A refracting surface can also serve as the stop through `lens.stopIndex`.
The analysis does not call `UpdateLens()` or mutate the prescription.

## Results

`result.surfaces` contains each surface's `sphericalBase`, `asphericDeparture`,
and combined `coefficients`. `result.totals` sums the combined values. Each
coefficient set exposes `S1` through `S5`, `AsTuple()`, and `AsDict()`:

| Term | Dictionary name | Meaning |
| --- | --- | --- |
| S1 | spherical | Primary spherical aberration |
| S2 | coma | Primary coma |
| S3 | astigmatism | Primary astigmatism |
| S4 | fieldCurvature | Petzval field-curvature term |
| S5 | distortion | Primary distortion |

These are unconverted Welford Seidel sums in mm. They are not transverse ray
errors, distortion percentages, or wavefront errors in waves. S4 is the Petzval
term; sagittal and tangential field curvature also involve S3. The coefficients
include the selected aperture and field scale. Doubling pupil radius multiplies
S1–S5 by `(16, 8, 4, 4, 2)`; doubling field multiplies them by `(1, 2, 4, 4, 8)`.

`result.paraxial` retains the wavelength, conjugate, aperture, field, optical
invariant, system transfer matrix, and surface histories. Each history contains
incident and outgoing marginal/chief `RayState` values (height, slope), indices
of refraction, vertex position, effective curvature, and quartic sag coefficient.
Positions are reconstructed from thicknesses relative to the first vertex.

## Even aspheres

The existing coefficient ordering remains `[A2, A4, A6, ...]`. To start with A4,
pass zero as the first entry. Both paraxial tracing and `PAEFL.SingleGroup()`
include A2 through effective vertex curvature:

```
c = 1/R + 2*A2
q = (1 + K)/(8*R**3) + A4
```

Seidel analysis compares the quartic sag `q` with the sphere of curvature `c`,
using departure `q - c**3/8`. This incorporates the change to the reference sphere
caused by A2. A6 and higher terms do not enter primary third-order aberrations.

The sums use the standard paraxial surface equations, with a division-free
distortion expression and height products for aspheric corrections. This avoids
singularities when marginal incidence or marginal height is zero. For formula
conventions, see the [RayOptics third-order analysis reference](https://ray-optics.readthedocs.io/en/stable/api/rayoptics.parax.thirdorder.html)
and [Ansys Seidel coefficient definitions](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v25101/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Seidel_Coefficients.html).

## Numerical backends

The analysis accepts NumPy or CuPy scalar lens parameters. Material RI calls
normalize scalar/list/array wavelengths to floating-point arrays on the selected
backend. The small transfer matrices, ray histories and Seidel coefficients
remain Python floats; reading a CuPy scalar synchronizes that value to the host.
There is no implicit conversion of a device array through NumPy.

Select the backend before importing the lens, material and surface modules:

```python
import Util.Backend as Backend
Backend.set_backend("CUDA")  # or "CPU"
from Lens import Lens
```

The framework's existing `from Util.Backend import backend as bd` imports bind
the backend at module import time. Switching it after those modules are loaded
does not update their bound references. The backend regression tests therefore
run CPU and CUDA analyses in separate processes.

## Validation

`tests/test_seidel.py` checks reference interface values, thick-lens EFL, medium
continuity through an embedded stop, chief-ray aiming, the optical invariant,
finite conjugates, aperture/field scaling, partial sums, A2 geometry and regular
zero-height/normal-incidence cases. Independent sag-intersection and exact Snell
traces verify the primary transverse ray errors for all five terms on spherical
and A2/A4 aspheric prescriptions. Zemax export parity has not been verified.

```powershell
python -m unittest discover -s tests -p test_seidel.py -v
python -m unittest discover -s tests -p test_seidel_backend.py -v
```

The backend tests compare real material models, all nine dispersion paths,
scalar and vector wavelengths, spherical and A2/A4 prescriptions, finite
conjugates, non-air environments, matrix/EFL calculations and partial sums.
CUDA testing requires CuPy and a working CUDA device; it is skipped when CuPy
is not installed.
