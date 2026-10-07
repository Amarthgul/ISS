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
  `fieldAngle` is ignored for finite objects unless `objectHeight=None`, which
  selects an angular chief-ray field at that distance. Heights and slopes refer to the
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

## Defocused objects with axial defocus excluded

Use `objectDistance` for the actual object and `focusDistance` for the object
distance defining the sensor's paraxial focus reference. Both are in mm,
measured back from the first vertex; infinity is supported. `focusDistance=None`
uses `objectDistance`, preserving the existing matched-conjugate calculation.
The added arguments are appended to the existing signatures, so existing
positional calls retain their meaning.

```python
# The current prescription must already be configured for focus at 1.35 m.
reference = lens.ComputeSeidelCoefficients(
    objectDistance=13500.0,
    focusDistance=1350.0,
    referenceField=10.0,
    referenceFieldUnits="degrees",
    stopSemiDiameter=3.0,
)
print(reference.totals.AsDict())  # Exactly the five Seidel wavefront coefficients
trace = reference.seidel.paraxial
print(trace.objectImageZ, trace.focusImageZ, trace.imageDefocus)  # mm
selected = reference.EvaluateField(-5.0)  # degrees with this normalization
```

`TraceLens()`, standalone and Lens `ComputeSeidel()`, and standalone and Lens
`ComputeSeidelCoefficients()` accept `focusDistance`. The prescription, pupil,
and imager are never changed. For a floating-element lens, configure its surface
thicknesses for the intended focus first; this analysis does not call
`BestFocusBFD()` or interpolate floating-element positions. The focus reference
is Gaussian/paraxial, not the sampled real-ray best-RMS focus used by rendering.

### Conjugates and the excluded term

Let the system matrix from the first to the last vertex be

$$
M=\begin{pmatrix}A & B \\ 
C & D \end{pmatrix}.
$$

An axial ray from object distance $d$ has initial slope $u_0=y_0/d$.
Its outgoing height and slope are $(A+B/d)y_0$ and $(C+D/d)y_0$.
Consequently the Gaussian image distance after the last vertex is

$$
L(d)=-\frac{A+B/d}{C+D/d},\qquad 1/\infty=0.
$$

`trace.ImageDistance(d)` evaluates this expression; omitting `d` uses the
actual object distance. A zero denominator represents an image at infinity.
Writing $z_N$ for the last vertex position, the recorded image planes are

$$
z_{\mathrm{object}}=z_N+L(d_{\mathrm{object}}),\qquad
z_{\mathrm{focus}}=z_N+L(d_{\mathrm{focus}}),\qquad
\Delta z=z_{\mathrm{focus}}-z_{\mathrm{object}}.
$$

These are `objectImageZ`, `focusImageZ`, and `imageDefocus`. Positive defocus
places the focus-reference plane farther toward positive Z. Matched conjugates
give zero defocus, including when their Gaussian image is at infinity. A finite
axial mismatch requires both image planes to be finite.

The axial focus mismatch supplies the quadratic pupil term $W_{020}\rho^2$.
The requested calculation excludes that term and evaluates the five Seidel
terms relative to the actual object's Gaussian conjugate, using the unchanged
surface equations below. No $W_{020}$ is added to `totals`, surface contributions,
partial sums, cumulative sums, or `EvaluateField()` results. All marginal/chief
rays and the invariant use **objectDistance**, not **focusDistance**.

Thus changing only `focusDistance` leaves all five coefficients unchanged;
changing `objectDistance` generally changes them, even at the same angular
field. This is the aberration content of the defocused zone, rather than a
transverse ray-error prediction at the displaced sensor. Spherical aberration,
astigmatism and field curvature remain intact. In particular, `W220` is retained:
it is field-dependent curvature, not the excluded axial focus offset. No fitted
Zernike defocus is subtracted from these conventional Seidel coefficients.

### Angular fields across object distances

`referenceFieldUnits=None` retains the existing convention: degrees at infinity
and signed object height in mm at finite conjugates. For finite conjugates the
other supported choice is `referenceFieldUnits="degrees"`. In that mode both
`referenceField` and `EvaluateField(field)` use the signed incident paraxial
chief-ray slope expressed in degrees, consistently with the infinite-object
API. No exact-angle tangent correction is introduced in this first-order model.

If the transfer matrix to the stop has first row $(a,b)$, the entrance pupil is
at $z_p=b/a$ relative to the first vertex. For angular reference slope
$\alpha=\textrm{radians}(\mathrm{referenceField})$, finite-object aiming is

$$
\bar u_0=\alpha,\qquad \bar y_0=-z_p\alpha,\qquad
H_{\mathrm{object}}=-(d_{\mathrm{object}}+z_p)\alpha.
$$

Here $H_{\mathrm{object}}$ is stored as `paraxial.objectHeight`; the minus sign
follows propagation toward positive Z. This angle is referenced to the entrance
pupil, rather than the first vertex, and scales object height with its distance
from that pupil. Supply `objectHeight=None` to `ComputeSeidel()` or `TraceLens()`
to obtain the same aiming directly with `fieldAngle`. In angular mode
`paraxial.fieldAngle` also retains the specified field for finite objects.
`EvaluateField()` scales both chief-ray histories and the derived object height
while retaining the focus and object conjugates.

For comparisons at constant angle of view, use the same angular reference field
and aperture normalization at every object distance. An explicit stop radius is
convenient when comparing a fixed physical aperture. This does not automatically
keep the same sensor framing if lens geometry or angular magnification changes.

## Calculation of the Seidel sums

The equations below describe `CalculateSeidel()` as implemented in
`src/Util/Analysis/Seidel.py`. Lowercase `s` denotes one surface's contribution;
uppercase `S` denotes a sum over surfaces. Every contribution uses the marginal
and chief rays traced through the **complete lens**, including when only a
subset of surfaces is summed.

### Paraxial ray histories

At surface `i`, the notation is:

| Symbol | Meaning | Units               |
| --- | --- |---------------------|
| $y_i, u_i$ | Incident marginal-ray height and slope | $mm$, dimensionless |
| $\bar y_i, \bar u_i$ | Incident chief-ray height and slope | $mm$, dimensionless |
| $u'_i, \bar u'_i$ | Outgoing marginal/chief slopes | dimensionless       |
| $n_i, n'_i$ | Incident and outgoing refractive indices at the selected wavelength | dimensionless       |
| $c_i$ | Effective vertex curvature, including A2 for an even asphere | $mm^{-1}$           |
| $t_i$ | Axial thickness following the surface | $mm$                |
| $q_i$ | Coefficient of $r^4$ in the surface sag expansion | $mm^{-3}$           |

Slope means `dy/dz`, represented by the paraxial angle in radians, rather than
the reduced slope `n*u`. Signed heights increase toward positive Y and rays
propagate toward positive Z. Refraction and translation follow

$$
u'_i = \frac{n_i}{n'_i}u_i
       + \frac{n_i-n'_i}{n'_i}c_i y_i,
\qquad
y_{i+1} = y_i + t_i u'_i.
$$

The chief ray follows the same equations with barred heights and slopes.
Refraction leaves height unchanged at the surface; the outgoing slope becomes
the next surface's incident slope after translation.

`TraceLens()` first computes the transfer matrix from the first vertex to the
stop. If its first row is `(a, b)`, the chief ray is aimed so that
`a*chiefHeight + b*chiefSlope = 0`. At infinity, its initial slope is
`radians(fieldAngle)` and its initial height is `-(b/a)*chiefSlope`; the marginal
ray has zero initial slope and is scaled by the chosen pupil or stop radius.
At finite conjugates, the chief ray starts at the specified object height and
the marginal ray starts on the object axis. Both are aimed and normalized using
the same stop matrix. Thus aperture and field normalization enter the equations
through the ray heights and slopes, rather than through a later scale factor.

### Spherical-base contributions

For each surface, calculate the marginal and chief incidence quantities and
the optical invariant:

$$
A_i = n_i(u_i+c_i y_i),
\qquad
\bar A_i = n_i(\bar u_i+c_i\bar y_i),
\qquad
H = n_i(y_i\bar u_i-\bar y_i u_i).
$$

`TraceLens()` evaluates `H` from the initial ray pair and retains it for every
surface. It is invariant under the paraxial refractions and translations.
Define the following differences, with outgoing values minus incident values:

$$
D_i = \frac{u'_i}{n'_i}-\frac{u_i}{n_i},
\qquad
P_i = c_i\left(\frac{1}{n'_i}-\frac{1}{n_i}\right),
\qquad
E_i = \frac{1}{(n'_i)^2}-\frac{1}{n_i^2}.
$$

The five contributions stored in `sphericalBase` are

$$
\begin{aligned}
s^{\mathrm{base}}_{1,i} &= -A_i^2 y_i D_i, \\
s^{\mathrm{base}}_{2,i} &= -A_i\bar A_i y_i D_i, \\
s^{\mathrm{base}}_{3,i} &= -\bar A_i^2 y_i D_i, \\
s^{\mathrm{base}}_{4,i} &= -H^2 P_i, \\
s^{\mathrm{base}}_{5,i} &= -\bar A_i\left[
    \bar A_i^2 y_i E_i
    -(2y_i\bar A_i-\bar y_i A_i)\bar y_i P_i
    \right].
\end{aligned}
$$

The distortion expression is the division-free form used in the code. It does
not divide by `A`, so normal marginal incidence remains regular. These signs
follow this implementation's Welford convention and ray-coordinate definitions.
For an asphere, `sphericalBase` refers to a sphere with the **effective vertex
curvature** `c`, rather than necessarily the prescription radius `R`.

### Aspheric contributions

The sag near the vertex is

$$
z(r) = \frac{c_i}{2}r^2 + q_i r^4 + O(r^6).
$$

A sphere with the same vertex curvature has quartic coefficient `c_i^3/8`.
The quartic departure and its Seidel scale are therefore

$$
d_i = q_i-\frac{c_i^3}{8},
\qquad
G_i = 8(n'_i-n_i)d_i.
$$

The contributions stored in `asphericDeparture` are

$$
\begin{aligned}
s^{\mathrm{asph}}_{1,i} &= G_i y_i^4, \\
s^{\mathrm{asph}}_{2,i} &= G_i y_i^3\bar y_i, \\
s^{\mathrm{asph}}_{3,i} &= G_i y_i^2\bar y_i^2, \\
s^{\mathrm{asph}}_{4,i} &= 0, \\
s^{\mathrm{asph}}_{5,i} &= G_i y_i\bar y_i^3.
\end{aligned}
$$

For a spherical surface, `d_i = 0`. For an even asphere, use
`c_i = 1/R_i + 2*A2_i` and `q_i = (1+K_i)/(8*R_i^3) + A4_i`, as described
under [Even aspheres](#even-aspheres). A2 changes both the paraxial rays and the
reference sphere used to calculate the departure. The height-product form
above avoids division by marginal height when it is zero.

### Surface, cumulative, and system sums

For each aberration `k = 1, ..., 5`, the combined surface contribution and the
inclusive cumulative sum through surface `j` are

$$
s_{k,i} = s^{\mathrm{base}}_{k,i}+s^{\mathrm{asph}}_{k,i},
\qquad
S_k^{(j)} = \sum_{i=0}^{j}s_{k,i}.
$$

For `N` surfaces, the system sum is $S_k=S_k^{(N-1)}$. An inclusive partial
sum is $S_k^{[a,b]}=\sum_{i=a}^{b}s_{k,i}$. The implementation uses
`math.fsum` to accumulate each of the five components.

`surfaces[i].coefficients` stores `s_{k,i}`, `CumulativeSums()[j]` stores
`S_k^(j)`, `totals` stores `S_k`, and `SumSurfaces(a, b)` stores the partial
sum. Partial sums retain the original full-system ray histories; they do not
retrace the selected surfaces as an isolated lens. A non-refracting stop has
`n' = n` and `u' = u`, so all five of its contributions vanish. Translation
through an air gap changes the ray heights used at the following surface but
adds no separate Seidel contribution.

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

### Calculation of the wavefront coefficients

`ComputeSeidelCoefficients()` traces at the chosen nonzero `referenceField`,
calculates the surface contributions and sums above, and then calls
`ToWavefront()`. At that reference field, `h = 1`. The conversion applies to
individual surface contributions, cumulative sums, partial sums, and system
totals alike:

| Wavefront coefficient | Single surface | System total | Polynomial term |
| --- | --- | --- | --- |
| $W_{040}$ | $w_{040,i}=s_{1,i}/8$ | $W_{040}=S_1/8$ | $W_{040}\rho^4$ |
| $W_{131}$ | $w_{131,i}=s_{2,i}/2$ | $W_{131}=S_2/2$ | $W_{131}h\rho^3\cos\phi$ |
| $W_{222}$ | $w_{222,i}=s_{3,i}/2$ | $W_{222}=S_3/2$ | $W_{222}h^2\rho^2\cos^2\phi$ |
| $W_{220}$ | $w_{220,i}=(s_{3,i}+s_{4,i})/4$ | $W_{220}=(S_3+S_4)/4$ | $W_{220}h^2\rho^2$ |
| $W_{311}$ | $w_{311,i}=s_{5,i}/2$ | $W_{311}=S_5/2$ | $W_{311}h^3\rho\cos\phi$ |

Because the conversion is linear, converting a cumulative Seidel sum gives the
same result as adding the converted surface contributions:

$$
W_{\alpha}^{(j)}=\sum_{i=0}^{j}w_{\alpha,i}.
$$

For example, the plotted spherical-aberration coefficient after surface `j` is

$$
W_{040}^{(j)}
=\frac{1}{8}\sum_{i=0}^{j}
\left[-A_i^2 y_i D_i+G_i y_i^4\right].
$$

The coefficient is the scalar $W_{040}^{(j)}$; multiplying it by $\rho^4$
evaluates its wavefront contribution at a particular normalized pupil radius.
All five coefficients have units of mm because both `rho` and `h` are
dimensionless. For wavefront values in waves, divide by the wavelength in mm,
`wavelength_nm * 1e-6`.

At another field, define `h = field/referenceField`. `EvaluateField()` scales
the chief-ray heights, chief-ray slopes, and invariant by `h`, leaving the
marginal ray unchanged. The selected-field Seidel sums then scale as

$$
(S_1,S_2,S_3,S_4,S_5)(h)
=\left(S_1,\ hS_2,\ h^2S_3,\ h^2S_4,\ h^3S_5\right)_{\mathrm{reference}}.
$$

The stored reference `W` coefficients remain unchanged. At `h = 0`, the four
off-axis polynomial terms vanish, even though their reference coefficients can
remain nonzero. Changing the reference field itself changes the normalization
and therefore scales `W131`, `W222`, `W220`, and `W311` by the corresponding
first, second, second, and third powers of the reference-field ratio.

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
