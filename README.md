# IOSS

Implicit Optical System Simulation. 

This project aims to establish a way to digitize lenses with physical accuracy for media production use, such as VFX live action matching or adding accurate optical characteristics in animated film.

Additionally, it may also be used to deconvolute images for information reconstruction, and in some cases, historical preservation of optics. 


# Functional Hierarchy 

The application is essentially divided into the following classes: 

- **_Imaging System_** 

  The overarching class that wraps all the subclasses and assembles them into an imaging system. 

  - **_Object space_** 

    This is the space in front of the lens. 

    - **_Point_** 

      A single point source, it is defined by field angle on x and y direction, and the distance d. This class is useful for testing lens performances and spot simulation. 

    - **_Image2D_** 

      A 2D color array, for example, an 8-bit image. Propagating the image through the lens could simulate what the image would be like when looked through the lens.

  - **_Lens_** 

    The lens class defines the lens. While this sentence sounds meaningless, it is to emphasize that other optical elements that may present in an imaging system, such as microlens, UV-IR cut, low-pass filter, etc., are not considered in the lens. The lens only defines the lens, as in “interchangeable lens camera”. 

    - **_Surface_**

      The key parameters of a surface include radius $r$, thickness $t$, clear semi-diameter $s_{d}$, and material. 
      Special surface types, such as even-aspheric and cylindrical surfaces may have additional parameters. 

    - **_Material_** 

      This class defines the optical material used by each surface. Each instance will search the material library and find the corresponding dispersion formula and coefficients, which will be used to calculate the index of refraction for each given wavelength. 

  - **_Imager_**

    The plane on which the image is formed. Currently it is a naive imager, in the future there might be additional UVIR elements and quantum efficiency in consideration. 
