# Fifth- and seventh-order aberrations

Third-order analysis describes the leading departure from ideal Gaussian imaging. Fifth and seventh orders describe how that departure changes as the aperture and field grow. They are especially useful when a lens balances opposing aberrations: a small third-order coefficient does not necessarily mean that rays near the pupil edge form a good image.

This analysis considers monochromatic, centered lenses with spherical or plane refracting surfaces. It uses the Gaussian image of the specified object as its reference, without moving the image plane to best focus. The aim is to explain how a prescription develops and corrects aberrations.

Two related quantities are distinguished throughout:

- **Transverse ray aberration:** displacement from the Gaussian image point, with additive contributions assigned to individual surfaces.
- **Wavefront aberration:** optical-path departure from a reference sphere. Both the complete-system coefficients and the cumulative diagrams are calculated from accumulated optical path. Each intermediate wavefront uses the Gaussian reference of the lens up to that surface.

## 1. Coordinates and the meaning of order

Let $\xi,\eta$ be dimensionless sagittal and tangential coordinates in the paraxial entrance pupil, and let $h$ be normalized field. Their polar form is

$$
\xi=\rho\sin\phi,\qquad \eta=\rho\cos\phi.
$$

Thus $\phi=0$ points along the positive tangential pupil direction. Physical ray positions $X,Y,Z$ are measured in millimeters. Ray slopes are

$$
u=\frac{dX}{dZ},\qquad v=\frac{dY}{dZ},
$$

and are distinct from both angles and direction cosines.

The starting ray is a linear combination of the paraxial marginal and chief rays. With marginal height/slope $y_m,u_m$ and reference-field chief height/slope $\bar y,\bar u$ at the first vertex,

$$
X=y_m\xi,\quad Y=y_m\eta+\bar y h,\qquad
u=u_m\xi,\quad v=u_m\eta+\bar u h.
$$

The chief ray is aimed at the stop paraxially. The marginal ray fixes the aperture normalization. This does not perform nonlinear aiming onto the real stop: the coordinates label a fixed paraxial pupil, even when pupil aberration changes the actual footprint.

At infinity, field is proportional to incident slope. If the physical incident angle is $\alpha$, then

$$
h=\frac{\tan\alpha}{\tan\alpha_{\rm ref}}.
$$

At finite conjugates, field is proportional to object height, $h=H/H_{\rm ref}$. The angular option defines that reference height from the paraxial chief-ray slope. The existing angle-in-degrees setting represents slope through $v_{\rm ref}=radians(\text{reference field})$; its physical angle is $\arctan v_{\rm ref}$.

A term $\xi^a\eta^b h^c$ has **total degree** $a+b+c$. Ray aberration in a centered system has the expansion

$$
\Delta\mathbf r
=\Delta\mathbf r^{(3)}
+\Delta\mathbf r^{(5)}
+\Delta\mathbf r^{(7)}+\cdots,
$$

after removing the first-order Gaussian image. Reversing both pupil and field reverses the transverse ray, so only odd total degrees occur. Wavefront terms have even degrees:

$$
W=W^{(4)}+W^{(6)}+W^{(8)}+\cdots.
$$

The order names refer to ray aberration. Fifth-order analysis therefore requires wavefront terms of degree six; seventh order requires degree eight.

## 2. Expanding the ray trace

Rather than deriving a separate long formula for each aberration, expand the surface-intersection and refraction equations in $\xi,\eta,h$. At every operation, discard monomials above the retained total degree. This is a local Taylor expansion about the axial ray, not a polynomial fit over sampled rays.

For example, square roots and reciprocal square roots follow

$$
(1+\epsilon)^{1/2}
=1+\frac{\epsilon}{2}-\frac{\epsilon^2}{8}
+\frac{\epsilon^3}{16}-\frac{5\epsilon^4}{128}+\cdots,
$$

$$
(1+\epsilon)^{-1/2}
=1-\frac{\epsilon}{2}+\frac{3\epsilon^2}{8}
-\frac{5\epsilon^3}{16}+\frac{35\epsilon^4}{128}+\cdots.
$$

When $\epsilon$ starts at degree two, its fourth power supplies degree-eight terms. Keeping these products is how the same tracing equations produce third, fifth and seventh orders.

### Intersection with a spherical surface

Measure local $Z$ from the surface vertex and let $c=1/R$ be signed curvature. The sphere near that vertex satisfies

$$
Z=\frac{c}{2}(X^2+Y^2+Z^2).
$$

Its familiar radial sag expansion is

$$
Z(r)=\frac{c}{2}r^2+\frac{c^3}{8}r^4
+\frac{c^5}{16}r^6+\frac{5c^7}{128}r^8+\cdots.
$$

The sextic and octic terms help explain why even a spherical surface has higher-order aberrations. Sag alone is not enough, however: the intersection moves laterally along an oblique ray, and Snell refraction is also nonlinear.

For a ray specified at the vertex plane, the hit point is

$$
X_s=X+uZ_s,\qquad Y_s=Y+vZ_s.
$$

Substitution into the sphere equation gives

$$
Z_s=\frac{c}{2}
\left[(X+uZ_s)^2+(Y+vZ_s)^2+Z_s^2\right].
$$

Starting with $Z_s=0$, repeated substitution determines successive degrees. Each iteration gains two degrees, so four substitutions determine the sag through degree eight. For a plane, $c=0$ and $Z_s=0$.

### Refraction

The incident unit direction and the surface normal oriented toward positive $Z$ are

$$
\mathbf d=\frac{(u,v,1)}{\sqrt{1+u^2+v^2}},
\qquad
\mathbf N=(-cX_s,-cY_s,1-cZ_s).
$$

The sphere equation makes $\mathbf N$ a unit vector. Let $\mu=\mathbf d\cdot\mathbf N$ and $\gamma=n/n'$, where $n,n'$ are the incident and outgoing refractive indices at the selected wavelength. The forward transmitted branch of vector Snell's law is

$$
\mathbf d'
=\gamma\mathbf d+
\left[\sqrt{1-\gamma^2(1-\mu^2)}-\gamma\mu\right]\mathbf N.
$$

Expanding this expression includes the angular nonlinearities of refraction. Convert back to slopes,

$$
u'=\frac{d'_x}{d'_z},\qquad v'=\frac{d'_y}{d'_z},
$$

and extrapolate to the same vertex plane:

$$
X'=X_s-u'Z_s,\qquad Y'=Y_s-v'Z_s.
$$

Both sides of the surface are now expressed at one plane. Translation by the following axial thickness $t$ is exactly linear in these coordinates:

$$
X_{\rm next}=X'+tu',\qquad Y_{\rm next}=Y'+tv'.
$$

This separates nonlinear surface effects from propagation between vertices.

### Why lower orders generate higher orders

A later surface receives an already aberrated ray. Its nonlinear response must be evaluated on that ray, not just on its paraxial part.

A one-variable illustration makes the coupling visible. Suppose an incoming coordinate is

$$
q(p)=p+a_3p^3+a_5p^5+\cdots,
$$

and a later surface has a cubic response $b_3q^3$. Then

$$
b_3q^3
=b_3p^3+3b_3a_3p^5
+3b_3(a_5+a_3^2)p^7+\cdots.
$$

A cubic response acting on a cubic incident error already produces fifth order. Seventh order includes cubic–quintic interactions and products of cubic errors. The full multivariate propagation retains these interactions automatically. Merely adding the $r^6$ or $r^8$ sag term to a Seidel formula would omit them.

## 3. Assigning contributions to surfaces

Write the ray state as $\mathbf q=(X,Y,u,v)$. Let $F_i$ be the complete Taylor map of surface $i$, and $L_i$ its first-order map:

$$
L_i\mathbf q=
\left(
X,\ Y,\
\frac{n}{n'}u-\frac{(n'-n)c}{n'}X,\
\frac{n}{n'}v-\frac{(n'-n)c}{n'}Y
\right).
$$

Define the surface's nonlinear increment on the actual incident polynomial ray:

$$
\mathbf e_i=F_i(\mathbf q_i)-L_i\mathbf q_i.
$$

Let $M_{i\rightarrow I}$ be the first-order transfer from just after that surface to the **final Gaussian image plane**. The transverse contribution is

$$
\Delta\mathbf r_i
=P\,M_{i\rightarrow I}\mathbf e_i,
$$

where $P$ selects the two position components. For example, if the first row of the remaining meridional transfer matrix is $(A_i,B_i)$, then

$$
\Delta X_i=A_i e_{X,i}+B_i e_{u,i},\qquad
\Delta Y_i=A_i e_{Y,i}+B_i e_{v,i}.
$$

The same formula applies separately to degrees three, five and seven. Because translations are linear, these increments telescope:

$$
\Delta\mathbf r^{(m)}
=\sum_i\Delta\mathbf r_i^{(m)},\qquad m=3,5,7.
$$

The cumulative correction through surface $j$ is

$$
\mathbf C_j^{(m)}=\sum_{i\le j}\Delta\mathbf r_i^{(m)}.
$$

Every entry uses the same final-image coordinates. It is not the spot size or the aberration of an independently refocused lens ending at surface $j$. Interactions are assigned to the surface receiving the aberrated ray. A non-refracting stop has $F_i=L_i$, so it adds zero; dividing an air gap with such a stop does not change the result. This additive ray decomposition remains available, but it is not used to generate the cumulative W tracks; those tracks use the combined optical-path wavefront described below.

## 4. Calculating the complete-system wavefront

Ray displacement and optical path contain related information, but the higher-order conversion between them depends on the pupil and reference geometry. The complete-system wavefront is therefore calculated directly from optical path.

### Accumulating optical path

A ray traveling an axial distance $t$ through index $n$ contributes

$$
\Delta S=n\,t\sqrt{1+u^2+v^2}.
$$

At a refracting surface, the signed trip from the vertex plane to the hit and back to the vertex plane contributes

$$
\Delta S_i
=Z_s\left[n\sqrt{1+u^2+v^2}
-n'\sqrt{1+u'^2+v'^2}\right].
$$

Signed extrapolation is essential: the vertex plane is a coordinate reference, not necessarily a physical boundary between the two media.

The incident phase must also be included. For an object at infinity, its plane wave gives, at the first vertex plane,

$$
S_0=n_0\,\frac{Xu+Yv}{\sqrt{1+u^2+v^2}}.
$$

For a finite object a positive distance $D$ before the first vertex,

$$
S_0=n_0D\left(\sqrt{1+u^2+v^2}-1\right).
$$

The subtracted constant sets an arbitrary phase origin. Similarly, constant axial path lengths may be dropped during propagation without changing aberration.

### Closing the path on the reference sphere

At the last vertex plane, let $\mathbf Q=(X,Y,0)$ be the outgoing ray position and $\mathbf d'$ its unit direction. The Gaussian image is

$$
\mathbf C=(0,Y_Gh,b),
$$

where $b\ne0$ is signed image distance from the last vertex and $Y_G$ is the reference-field Gaussian image height. Choose the reference sphere centered at $\mathbf C$ and passing through the last axial vertex. Its radius squared is

$$
R_{\rm ref}^2=b^2+(Y_Gh)^2.
$$

Set

$$
a=(\mathbf Q-\mathbf C)\cdot\mathbf d',\qquad
d=|\mathbf Q-\mathbf C|^2-R_{\rm ref}^2.
$$

The signed distance along the ray to the near-vertex branch of that sphere is

$$
\ell=-a-sgn(b)\sqrt{a^2-d}.
$$

This choice gives $\ell=0$ for the axial ray and handles real and virtual Gaussian images. Intersecting the outgoing ray with the sphere matters: replacing this step by a straight distance from $\mathbf Q$ to the image point changes higher-order terms.

If $S$ is the accumulated path to the last vertex plane, define

$$
A(\xi,\eta,h)=S+n_I\ell.
$$

With the sign convention used here, wavefront aberration is reference path minus actual path. Removing the pupil-independent phase gives

$$
W(\xi,\eta,h)
=-\left[A(\xi,\eta,h)-A(0,0,h)\right],
$$

from which degrees four, six and eight are retained. No best-fit sphere or sensor refocus is introduced. The expansion remains parameterized by the input paraxial pupil, rather than a nonlinearly remapped exit-pupil grid.

### Resolving the W coefficients

For rotational symmetry, write

$$
W=\sum_{i,j,k}W_{ijk}\,h^i\rho^j\cos^k\phi.
$$

Here $i$ is field degree, $j$ is pupil degree, and $k$ is the angular power. The Cartesian form of each basis function is

$$
h^i\rho^j\cos^k\phi
=h^i(\xi^2+\eta^2)^{(j-k)/2}\eta^k.
$$

The allowed $k$ values have the same parity as $i,j$ and satisfy $0\le k\le\min(i,j)$. Matching coefficients of the Cartesian polynomial therefore separates the angular dependence without fitting rays. Field-only piston is excluded.

| Ray order | Wavefront degree | Independent coefficients |
| --- | --- | --- |
| 3 | 4 | W040, W131, W220, W222, W311 |
| 5 | 6 | W060, W151, W240, W242, W331, W333, W420, W422, W511 |
| 7 | 8 | W080, W171, W260, W262, W351, W353, W440, W442, W444, W531, W533, W620, W622, W711 |

For example, the complete fifth-order wavefront is

$$
\begin{aligned}
W^{(6)}={}&W_{060}\rho^6+W_{151}h\rho^5\cos\phi\\
&+h^2\rho^4(W_{240}+W_{242}\cos^2\phi)\\
&+h^3\rho^3(W_{331}\cos\phi+W_{333}\cos^3\phi)\\
&+h^4\rho^2(W_{420}+W_{422}\cos^2\phi)\\
&+W_{511}h^5\rho\cos\phi.
\end{aligned}
$$

W060 and W080 are the on-axis higher-order spherical terms. Away from the axis, different pupil and angular powers distinguish corrections that cannot be described by simply extending the five primary Seidel numbers.

All W values are in millimeters because pupil and field coordinates are dimensionless. To express them in waves, divide by the wavelength in millimeters. At third order, this convention reproduces the wavefront coefficients described in [Seidel analysis](Seidel.md).

## 5. Combined cumulative wavefronts

The cumulative tracks apply the optical-path calculation of Section 4 after **each surface**, using the actual Taylor ray and optical path already accumulated through the preceding lens. They do not convert transverse ray errors into wavefront coefficients.

The input ray family is fixed once for the complete lens. Every intermediate calculation retains the same object, wavelength, pupil normalization and reference field. In particular, the rays are not re-aimed at a different stop when an intermediate wavefront is evaluated.

### The reference for a partial system

Let $y_{m,j},u_{m,j}$ be the outgoing paraxial marginal height and slope at surface $j$, and let $\bar y_j,\bar u_j$ be its outgoing reference-field chief height and slope. The Gaussian image of the partial system lies at

$$
b_j=-\frac{y_{m,j}}{u_{m,j}},\qquad
Y_{G,j}=\bar y_j+b_j\bar u_j.
$$

Here $b_j$ is measured from that surface's vertex, not from the final lens vertex. Use the sphere centered at

$$
\mathbf C_j=(0,Y_{G,j}h,b_j)
$$

and passing through vertex $j$. Positive and negative $b_j$ select real and virtual Gaussian images. The signed sphere-intersection calculation from Section 4 gives $\ell_j$, the distance from the actual outgoing ray's vertex-plane position to that reference sphere.

If $S_j$ is the accumulated optical path through surface $j$, form

$$
A_j(\xi,\eta,h)=S_j+n'_j\ell_j,
$$

$$
W_j(\xi,\eta,h)
=-\left[A_j(\xi,\eta,h)-A_j(0,0,h)\right].
$$

Extracting degrees four, six and eight gives the combined coefficients $W_{ijk}^{[j]}$. These are optical-path Taylor coefficients of the partial system with the original input ray family. All interactions with earlier surfaces are retained in both the ray and its accumulated path.

The reference closure is only an evaluation of the current wavefront. It is not added to the path propagated toward the next surface. The next calculation continues from $S_j$, adding the physical gap and surface paths, and closes on its own reference.

### Collimated intermediate bundles

A partial system may be afocal even when the complete lens forms a finite image. If $u_{m,j}=0$, its Gaussian reference is a plane rather than a sphere.

Set $v_{G,j}=h\bar u_j$. The plane through the vertex has normal proportional to $(0,v_{G,j},1)$. For an actual outgoing ray $(X,Y,u,v)$, its signed distance to the plane is

$$
\ell_j=
-\frac{Yv_{G,j}\sqrt{1+u^2+v^2}}{1+v\,v_{G,j}}.
$$

The same expression $A_j=S_j+n'_j\ell_j$ then supplies the wavefront. This handles initial stops and plane surfaces without inventing a distant finite focus. A finite partial-system image exactly at its own vertex is outside this reference convention: the corresponding axial reference sphere has zero radius.

### Relation to the complete-system result

At the last surface, the partial system is the complete lens. The reference and accumulated optical path are the same, so

$$
W_{ijk}^{[N]}=W_{ijk}^{\rm system}.
$$

This equality holds directly within the retained Taylor order. It is not obtained by rescaling bars, distributing a final residual, or adjusting their signs. The earlier paraxial ray-to-wavefront projection has been removed. There is no additional wavefront-conversion approximation beyond the Taylor truncation and the stated optical model.

Differences can be written as

$$
\delta W_{ijk}^{[j]}
=W_{ijk}^{[j]}-W_{ijk}^{[j-1]},
$$

but these are **changes in the combined wavefront under successive Gaussian references**, not an isolated, reference-independent aberration generated by surface $j$. The partial-system Gaussian image and reference sphere generally change with each surface. Such differences include refraction, propagation, and the change of reference. They should not be confused with the additive transverse-ray contributions from Section 3 or with classical higher-order surface sums.

## 6. Reading the correction diagrams

Each bar is the combined coefficient after that surface. A coefficient moving toward zero means the accumulated wavefront is closer to its current Gaussian reference in that particular term. A change of sign is a sign change of the optical-path coefficient under that reference; it is no longer a sign produced by the retired ray-to-wavefront approximation.

For comparing the relative effect of surfaces, keep in mind that adjacent bars use different reference spheres. A large jump can reflect a change in the partial-system conjugate as well as a refractive correction. References near an intermediate Gaussian focus can be especially sensitive. Matching the complete-system value does not, by itself, prove that every intermediate difference is a unique physical surface contribution.

A bar extends over the following axial thickness for visual alignment with the lens. Its value is evaluated immediately after the surface, before that gap. The next bar includes propagation through the gap.

A non-refracting stop adds no refractive error, but its wavefront bar need not equal the preceding bar: propagation and the sphere passing through the new vertex change the reference. A coincident stop in the same medium, with zero preceding gap, leaves both wavefront and reference unchanged. Inserting a non-refracting stop in a gap does not alter the downstream physical system or its final coefficients. Zero-thickness surfaces still affect subsequent values even when their own bars have no width.

Coefficients of different orders must be evaluated with their pupil and field powers before judging their relative importance. On axis,

$$
W(\rho,0)=W_{040}\rho^4+W_{060}\rho^6+W_{080}\rho^8+\cdots.
$$

At half pupil radius, these terms are reduced by factors of $16,64,256$, respectively. Higher orders can strongly affect the outer pupil while having little effect near its center. Off axis, the field and angular factors matter as well; a zero at one azimuth is not correction over the entire pupil.

Changing the normalization also changes the numerical coefficients. Increasing the reference pupil radius by a factor $s$, with the same physical lens and field reference, scales Wijk by $s^j$. Increasing reference field slope or object height by a factor $t$ scales it by $t^i$. Comparisons therefore need the same wavelength, conjugates, pupil and field conventions.

These are local expansions. Seventh order can improve the prediction without making a wide-field or fast-aperture result uniformly accurate. A small coefficient may result from cancellation rather than rapid convergence, and higher orders can remain important. The model also excludes vignetting, nonlinear stop aiming, diffraction, aspheres, decentered surfaces and mirrors. The complete-system image reference must be finite and distinct from the last vertex. Intermediate references may be spherical or planar; finite intermediate images must be distinct from their evaluation vertices. Taylor truncation and numerical errors remain, so eliminating the conversion error is not a guarantee of full-aperture accuracy.

## Further reading and use

Taylor ray maps provide a natural extension of matrix optics to nonlinear lens behavior; see Hullin, Hanika and Heidrich, [Polynomial Optics: A Construction Kit for Efficient Ray-Tracing of Lens Systems](https://diglib.eg.org/items/71d1de0c-134d-48f1-ba75-684f65c55019) (2012). The equations above specify this project's particular coordinates, surface attribution and reference-wavefront convention.

For practical use, [HigherOrder.py](../src/Util/Analysis/HigherOrder.py) returns the additive surface ray contributions, combined optical-path W through each surface, and complete-system W coefficients. The cumulative wavefront sequence is available through CumulativeWavefrontCoefficients; CumulativeCoefficients retains its original meaning as cumulative transverse ray errors. [SurfaceData.py](../src/Util/Analysis/SurfaceData.py) selects the combined wavefront tracks through displayConfigHigherOrder. Both the plotted final values and the reported system wavefront use the same optical-path calculation.
