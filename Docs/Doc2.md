# 2 - Geometric Optics

Introductory courses in optics commonly uses the paraxial approximation and treat all lenses as thin lenses with no thickness. Rays are represented with a height from the optical axis and an angle. 

However, paraxial approximations can not be taken for granted when the goal is to accurately simulated the imaging process. For example, in most paraxial cases, only the marginal and chief ray needs to be considered, even other calculations will limit the ray to be within the tangential plane. 

The geometric optics part is based on a representation of ray: 

$$
\mathbf{r}=\left(  x,\\ y, \\ z, \\ v _x, \\ v _y, \\ v _z, \\ \lambda, \\ \Phi, \\ i _{\Phi}, \\ b, \\ s, \\ C, \\ AOV \right) ^T 
$$

In those components:

- $x$, $y$, and $z$ are the location of the base point of rays, typically their position on a surface. $v _x$, $v _y$ and $v _z$ are the vector directions. These 2 sets of parameters can be used to express the **position and direction** of rays.
    
    For the ray transfer matrix used in paraxial systems, rays are typically represented with only the height $h$ and angle $\gamma$. This cannot be used here due to the fact that the optical system may not be axisymmetric, and the rays quite often do not reside in the tangential plane.
    
- $\lambda$ is the wavelength, which is used to calculate the refraction index.
    
    Note that, while the wavelength in here is described in nanometers, many of the calculations are treating the wavelength in micrometers, such as the dispersion formulas.
    
- $\Phi$ is one of the diagonal elements in the quadratic form of the polarized radiance ellipse. In section 2.2.3 it is written as $a$.
- $i _{\Phi}$ is the other diagonal element in the quadratic form of the polarized radiance ellipse. In section 2.2.3  it is written as $c$.
- $b$ is angle component used to in the quadratic form of the polarized radiance ellipse. This combined with the other 2 terms can be used to calculate the radiance of the ray on a given pair of $s$ and $p$ polarization direction. More on this can be seen in section 2.2.3.
- $s$ is an index denoting after which **s**urface this ray is currently located. 
For example, if a ray just interacted with surface $i$ and refracted through it, its index will be updated to $i$. This can help determining the relative location of the ray more easily than using the position and direction.
- $C$ is an optional parameter for color channel. When set to something not -1, it corresponds one of the RGB channels, and during wavelength to RGB conversion it will contribute only to that channel.
- $AOV$ refers to “another other value”. It is another optional parameter and may be in itself an array. This parameter is purely for carrying infomation of the pixel for post production purposes.

After the terms above, there could be further data concatenated at the back. These data, however, will not be used for calculation, they are merely render pass data that needs to be transferred from the object space to the image space, such as motion vector, object-ID, surface normal, etc. 

At a higher level, this framework is modelled based on referencing the imaging equation developed by H. H. Hopkins:

$$
I \left(x, \, y \right) = \left ( \frac{1}{f \lambda}  \right ) \int \int _{\infty } \sigma \left ( x_0, \ y_0 \right )
\left| s \left ( x, \ y \right )t \left ( x, \ y \right ) ** psf \left ( x, \ y \right ) \right| ^2 \mathrm{d} x_0 \textrm{d} y_0
$$

In the equation above, $I \left(x, \, y \right)$ represents the image, where x and y can be thought of as pixel location. $\sigma \left ( x_0, \ y_0 \right )$ is the illumination; $s \left ( x, \ y \right )$ is the optical field, $t \left ( x, \ y \right )$ is the transmittance, and $psf$  is of course the point spread function. For incoherent illumination, this equation would become effectively a large convolution using the $psf$. 

In a way, common computer graphics lens effect also uses this equation to some extents (the rendering equation), but with two big simplifications. For one, they largely ignores the $\lambda$ term, i.e., the wavelength. And second, they removed the $x$ and $y$ term in the equation, forcing the imaging process to become field-irrelevant, which is why the bokeh are identical even at different focus distances, field angles, and aperture settings. 

This framework aims to solve the lack of field angle and wavelength in current computer graphics rendering model while keeping it compatible with the current media production pipeline, so that accurate imaging results can be created without drastic workflow or budget change. 
