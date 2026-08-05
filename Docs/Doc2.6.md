# 2.6 - Imager

The imager is sometimes also called a detector, but regardless of the name, it refers to the component in the imaging system tasked to gather lights (or other forms of electromagnetic waves) and convert it into an image. 

# 2.6.1 - Plain imager

A plain imager is an ideal plane that simply intercepts and takes an integral over all incident rays. 

Recall that the ray definition is:

$$
\mathbf{r}=\left(  x,\\ y, \\ z, \\ v _x, \\ v _y, \\ v _z, \\ \lambda, \\ \Phi, \\ i _{\Phi}, \\ b, \\ s, \\ C, \\ AOV \right) ^T 
$$

The position and direction of the ray can be used to locate the place where the ray falls at on the plane and map that 3D position to a pixel on an image. a very trivial idea. 

There are two ways of reading the radiance info of the ray. The first is to simply read the $\Phi$ term, which is suitable when the system is set to the basic radiometry model. However, if polarization is enabled, reading only the first term will be inaccurate and may create unevenness on the horizontal and vertical direction of the image. 

The second way would be to use the combined terms $\Phi$, $i_ {\Phi}$, and $b$ to construct the polarization ellipse, then use the ellipse to get the radiance. Detailed ways on how to construct the polarization ellipse is described in previous chapter 2.2.3. 

After getting the radiance, the next step would be to distribute the radiance into one or more color channels. This framework uses another two ways to perform such allocation, depending on how rays on the object side are generated.

If the rays are generated using Fraunhofer line replacement and interpolation, it means that the rays only have a handful of unique wavelengths. In this case, it is more efficient to pre-calculate a per-channel weight for each unique wavelengths, then allocate the ray’s radiance accordingly.

If, on the other hand, the wavelengths are generated with probability density function, then with millions of unique wavelengths, the same pre-calculated weight will generate an overhead more consuming than the rest of the system combined. The work around, however, is surprisingly easy. 

As described in the initial ray structure, an optional $C$ term is reserved for channel information. By the design contract, when the probability density function method is adapted during emission, the $C$ term will be filled with a designated channel index. By default, 0 for red, 1 for green, and 2 for blue. So, when the ray reaches the imager, there is no need to decompose it into RGB channels; the framework only looks at the channel index $C$ and directly deposit the radiance into the channel indicated by the index. 

It may occur to some that this method ignored how wavelengths often converts to more than one channel of color. While this intuition is true, the PDF are not entirely isolated, but with some overlaps. So it is quite often the case where two rays carry nearly the same wavelength but different color channel index. In this case, the two rays contribute to two different channels despite having roughly the same wavelength, effectively replicating the weighing process. 

The figure below shows an example of the color PDF and their aggregate. It can be seen that while each channel has a peak, they overlap with each other and creates a smoother transition in-between. This overlap could thus help creating the secondary colors. 

<p align="center">
	<img src="../resources/ReadmeImg/Doc2.6/PDF.png" width="640">
</p>

While named "plain", the plain imager is only plain in the sense that it has no other modifiers outside or inside of it, i.e., rays are intersected and radiance are deposit directly without modifications. 
There are still things that the plain imager is capable of. For example, some old film point and shot cameras have a curved image plane, which aims to solve the field curvature caused by the cheap lens. The plain imager is nevertheless a surface, which means it could inherit the behaviors of a lens surface and have curvature of its own. In this way, the imager itself can also be curved. 

# 2.6.2 - CMOS/CCD

Digital sensors are mostly divided into two categories, CMOS and CCD, but their difference primarily lies in how they read the data. CCD does have some unique image characteristics such as blooming, but its logic also made it near impossible for motion picture production, so it can be ignored temporarily for this framework. 

Since silicon based PDA has a spectral sensitivity up to 1100nm, and the framework technically has the ability to produce wavelengths in UV and IR range, there is a non-zero chance that non-visible lights will be recorded. For this purpose, it is recommended to include a bandpass filter that culls rays beyond the visible spectrum range.   

# 2.6.3 - Film

Film is similar to a plain imager, but it has additional grain and spectral response settings, if necessary, the effect of halation can also be added. 

The spectral response by default is defined using the same skewed Gaussian distribution as used in the object space probability density function:

$$
f\left( x \right)= g \cdot \frac{2}{\sqrt{2 \pi \sigma ^2}}e ^{- \frac{\left ( x-\mu \right ) ^2}{2}} \left [ 1+ \textrm{erf}\left ( \frac{\alpha x}{\sqrt{2}} \right ) \right ]
$$

However, while in object space, the gain coefficient $g$ and the skewness coefficient $\alpha$ are often left unused or used only at default values, the film spectral response approximation often used all of these terms. Results from invoking one of these spectral responses are also not used as a probability, but as a multiplier that scales the radiance. 

If the object space emission is using the same probability density function as the spectral response of the film detector, then the resulting color will be a one to one replication (not including the aberrations), with no color shifts or white balance difference. But any difference between the two will create a shift of color. For example, if the blue response received a boost in gain, then the resulting image will have a blue tint. This would be the case for many tungsten balanced film, as is the case for Kodak Vision-3 5219 500T, as shown in the figure below. 

<p align="center">
	<img src="../resources/ReadmeImg/Doc2.6/Spec.png" width="540">
</p>

Note that 5219 is a color negative film, whose RGB color is represented by emulsion layers with dyes of their opposite color. That is why the layers are written as yellow-forming, magenta-forming, and cyan-forming, instead of blue, green, and red.