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

## Field-independent coefficients and cumulative correction

Use `lens.ComputeSeidelCoefficients()` (or the standalone function in
`Util.Analysis.Seidel`) to calculate the coefficients of a combined field/pupil
wavefront polynomial. It uses a fixed, nonzero reference field and defines
`h = field / referenceField`. Coefficients depend on the aperture, wavelength,
conjugate and chosen reference normalization, but do not change when evaluating
a different field, including zero.

```python
reference = lens.ComputeSeidelCoefficients(referenceField=10.0)

# Each physical surface's field-independent wavefront coefficients in mm.
for row in reference.surfaces:
    print(row.surfaceIndex, row.coefficients.AsDict())

# Suitable for a plot of how spherical aberration is corrected through the lens.
surfaceIndices = [row.surfaceIndex for row in reference.surfaces]
saContributions = [row.coefficients.W040 for row in reference.surfaces]
saCumulative = [row.W040 for row in reference.CumulativeCoefficients()]

# Reference-field Seidel sums are retained alongside the wavefront coefficients.
print(reference.seidel.totals.AsDict())
print(reference.SumSurfaces(first=2, last=5).AsDict())

# Selected-field sums with matching chief-ray histories; no new lens trace.
onAxis = reference.EvaluateField(0.0)
offAxis = reference.EvaluateField(20.0)
```

The polynomial is

```
W = W040*rho^4 + W131*h*rho^3*cos(phi)
  + W222*h^2*rho^2*cos(phi)^2 + W220*h^2*rho^2
  + W311*h^3*rho*cos(phi)
```

`rho` is normalized pupil radius. The values are coefficients, not the complete
evaluated polynomial terms, and are in mm rather than waves. The conversions
from reference-field Seidel sums are `W040=S1/8`, `W131=S2/2`, `W222=S3/2`,
`W220=(S3+S4)/4`, and `W311=S5/2`. W220 includes the astigmatic contribution;
the separate Petzval term is S4/4. `SeidelCoefficients.ToWavefront()` performs
these conversions for any selected-field sum, but those converted amplitudes
still include that selected field's powers. The reference API supplies the
field-independent polynomial interpretation.

`referenceField` defaults to 1 degree at infinity or 1 mm of object height for
a finite `objectDistance`. It must be nonzero. `reference.fieldUnits` records
the units to use with `EvaluateField()`. At finite conjugates, for example:

```python
reference = lens.ComputeSeidelCoefficients(objectDistance=800.0, referenceField=35.0)
selected = reference.EvaluateField(0.0)  # zero object height, in mm
```

`reference.surfaces` retains spherical-base and aspheric-departure coefficients
separately. `reference.CumulativeCoefficients()` and
`reference.seidel.CumulativeSums()` return inclusive running totals aligned with
the surface order. Their final entries equal the corresponding system totals;
every contribution is derived from the same full-system rays.

## Surface-data plotting

The Seidel tracks use the same aligned subplots and bars as the existing
prescription tracks. They plot cumulative wavefront coefficients, not individual
surface contributions or selected-field Seidel sums.

```python
from Util.Analysis.SurfaceData import SurfaceDataType, displayConfigSeidel

fig, axes = lens.PlotSurfaceData(
    DisplayConfig=displayConfigSeidel,
    SeidelReferenceField=10.0,
    PlotTrackLength=100.0,
)

# Mix any of the five Seidel types with existing data tracks.
fig, axes = lens.PlotSurfaceData(DisplayConfig=[
    SurfaceDataType.OpticalPower,
    SurfaceDataType.RefractiveIndex,
    SurfaceDataType.AbbeNumber,
    SurfaceDataType.SeidelW040,
])
```

Omitting `DisplayConfig` retains the module's `displayConfig` selection. The
five types are `SeidelW040`, `SeidelW131`, `SeidelW222`, `SeidelW220`, and
`SeidelW311`; each adds one track. All selected Seidel tracks share one reference
calculation. `SeidelReferenceField` defaults to 1 degree and is recorded in the
figure title; wavelength is the surface-data plot's d line (587.56 nm).

Bar height is the inclusive cumulative coefficient after that surface. The bar
extends from its vertex to the next vertex, or over the final surface's prescribed
thickness. Air gaps and stops retain the accumulated value. Zero-width spans
produce no bar, but their contributions remain in subsequent cumulative values.
Seidel axes include zero, preserve signed heights, and use significant-digit
labels to keep small coefficients visible. Values are in mm at normalized field
`h = field / SeidelReferenceField`.

The figure starts at 16 by 10 inches and grows vertically when needed to reserve
one inch of plotting height for each data track. The lens retains equal physical
scale and the same horizontal alignment as the coefficient and material tracks.
The data tracks follow the layout's displayed width on every redraw, preserving
axial alignment when the plot window is resized or the figure is exported.

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
