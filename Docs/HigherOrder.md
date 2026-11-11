# Fifth- and seventh-order aberrations

`lens.ComputeHigherOrder()` calculates monochromatic aberrations for centered
vintage-style spherical lenses. Plane refracting surfaces and non-refracting
stops are supported. The calculation uses truncated Taylor algebra, not a fit
to sampled rays. It does not update the lens, move surfaces, trace reflections,
apply clipping, or change existing Seidel results. No plotting API is added.

```python
result = lens.ComputeHigherOrder(
    maxOrder=7,                 # 3, 5, or 7; includes all lower odd orders
    referenceField=10.0,
    pupilSemiDiameter=5.0,      # mm; alternatively use stopSemiDiameter
)

# Full-system wavefront coefficients, in mm rather than waves.
print(result.wavefront.ForOrder(5).AsDict())  # W060, W151, ... W511
print(result.wavefront.ForOrder(7).AsDict())  # W080, W171, ... W711

# Each surface's fifth-order transverse image-error polynomial.
for row in result.surfaces:
    print(row.surfaceIndex, row.coefficients.ForOrder(5).AsDict())

# Tangential pupil edge at the reference field: (dx, dy) in mm.
print(result.totals.ForOrder(7).Evaluate(0.0, 1.0, h=1.0))
print(result.totals.Evaluate(0.0, 1.0, h=1.0))  # third + fifth + seventh

# Ready for future contribution and cumulative diagrams.
cumulative = result.CumulativeCoefficients()
rearGroup = result.SumSurfaces(first=3)
```

The standalone function is `Util.Analysis.HigherOrder.ComputeHigherOrder`.
The supported contract is a nonempty centered sequential spherical/plane
refractive prescription with positive indices, a defined nonsingular pupil,
and a finite Gaussian image at a nonzero distance from the last vertex.
Aspheres, transformed/decentered surfaces, mirrors and afocal systems are
outside this implementation's contract.

## Coordinates and normalization

`pupilX` and `pupilY` are dimensionless coordinates at the **paraxial entrance
pupil**, with y in the field's meridional plane. `h` is normalized field.
Ray-polynomial dictionary keys `(a, b, c)` mean `pupilX**a * pupilY**b * h**c`.
Missing monomials are zero. Both components have units of mm at the Gaussian
image plane; distortion remains included relative to the paraxial image.

At infinity, `referenceField` is an incident slope expressed in degrees,
matching `TraceLens` and `ComputeSeidelCoefficients`: the exact input slope
is `h * radians(referenceField)`, and its physical angle is `atan(slope)`.
For an intended physical reference angle alpha, pass
`referenceField=degrees(tan(radians(alpha)))`. At another physical angle beta,
use `h=tan(radians(beta))/tan(radians(alpha))`. This keeps the expansion
coordinate linear without silently changing existing Seidel conventions.

At finite `objectDistance` (positive mm back from the first vertex), the
default reference field is object height in mm. `referenceFieldUnits="degrees"`
instead derives that height using the paraxial chief-ray slope. `h` then scales
the derived object height. `referenceField` is nonzero in either mode.

The default pupil radius is `lens.entrancePupil.clearSemiDiameter`. An explicit
`stopSemiDiameter` sets the first-order marginal-ray normalization at the stop.
Neither option performs nonlinear ray aiming or ensures an exact real-ray
footprint at the stop. A prescription stop contributes zero aberration itself.
This pupil convention must be kept fixed when comparing orders or lenses.

The image reference is the actual object's Gaussian image; no best-focus or
sensor-defocus correction is added. `result.reference.imageDistance` is relative
to the last vertex and `imageHeight` is the reference-field Gaussian height.
`result.paraxial` retains the first-order histories and normalization.

## What each surface contribution means

A spherical surface is traced from its vertex plane to its actual intersection,
refracted using vector Snell's law, then extrapolated back to its vertex plane.
Ray slopes are exact `dx/dz, dy/dz`; direction vectors are normalized for Snell.
Translation between vertex planes is therefore linear in the ray state.

For an incident polynomial ray state q, let F_i be this full surface map and
L_i its first-order map. The nonlinear increment is `F_i(q) - L_i(q)`.
Transfer this increment through the remaining first-order system to the common
Gaussian image. Its position components are the surface's contribution.

The input q contains all preceding nonlinear changes, so fifth- and seventh-
order interactions are included and attributed to the surface receiving that
aberrated ray. The increments telescope: their sum is the complete transverse
Taylor image error. Cumulative entries are in common final-image coordinates,
not the local spot size or wavefront after a truncated lens prefix. Adding an
unpowered stop or splitting an air gap does not change the total.

**These are ray coefficients, not classical surface W coefficients.**
`result.totals` and every `surfaces[i].coefficients` have the same ray convention.
`result.wavefront` is a separate full-system result. In particular, integrating
seventh-order image error with only the paraxial ray/wavefront conversion would
miss higher-order coordinate corrections. The implementation does not label
that approximation as a surface wavefront decomposition.

## Wavefront coefficients

`Wijk` multiplies `h**i * rho**j * cos(phi)**k`, where
`pupilX=rho*sin(phi)` and `pupilY=rho*cos(phi)`. Values are mm; divide by
`wavelength_nm * 1e-6` to obtain waves. `Evaluate(rho, phi, h)` uses phi in radians.

| Ray order | Wavefront degree | Coefficients |
| --- | --- | --- |
| 3 | 4 | W040, W131, W220, W222, W311 |
| 5 | 6 | W060, W151, W240, W242, W331, W333, W420, W422, W511 |
| 7 | 8 | W080, W171, W260, W262, W351, W353, W440, W442, W444, W531, W533, W620, W622, W711 |

Signed optical path is accumulated along the ray, including vertex-plane
extrapolations. The outgoing ray is intersected with a reference sphere centered
on the Gaussian image and passing through the last axial vertex. Positive W
means reference optical path minus actual optical path, matching the existing
Seidel wavefront sign. Field-only piston and terms below degree four are omitted.
Real and virtual Gaussian images use the corresponding signed sphere branch.

The series is expressed in input pupil coordinates, rather than reparameterized
onto a nonlinear exit-pupil grid. No Zernike fitting or best-fit sphere is used.
At third order, the resulting W coefficients reproduce the existing Seidel
coefficients under the same pupil, field and conjugate normalization.

## Implementation and useful limits

`HigherOrder.py` handles spherical propagation and surface contributions.
`RayReference.py` provides small CPU float64 polynomials, reference launch and
wavefront extraction. Scalar prescription values can originate on either backend;
the coefficient algebra stays on the CPU. Spheres use their implicit equation,
so no new sag-series interface or changes to the rendering tracer are required.

This is an educational local expansion around the axial ray. Adding seventh
order generally improves small-field/aperture predictions, but convergence is
not guaranteed at a wide field or fast aperture. Opposing corrections can also
make one order small without making subsequent orders small. The terms show
how the prescription balances corrections; they are not a substitute for
full-aperture ray tracing.

`tests/test_higher_order.py` compares with existing Seidel coefficients and an
independent exact spherical ray/OPL trace, and checks conjugates, virtual images,
plane refraction, stop invariance, normalization scaling and surface sums.
