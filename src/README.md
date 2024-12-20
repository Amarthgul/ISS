

# TODO

- Add memory estimation for image sim, recurse the process if there are more entries than the system can handle. 

- Reflection 

  - Add line separation to create the inner barrel 

  - Add TIR implemetation.

  - Add diffuse when hitting inner barrel 

- Transmission lose 

- Wave optics 




# Code Conventions

## Underscore 

Single underscore methods means they are private to the class and is required for the class to perform its duty. Double underscore methods are for internal testing purpose only and typically not used when the class is called externally. 

For example, the method `ellipsePeripheral()` appeared both in the `Lens` class and in the `ImagingSystem` class. However, in the `Lens` class it is named `__ellipsePeripheral()`, as it is only used for testing the lens class during development and not a part of the functionality for the `Lens` class by contract.  



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

    
# Record Keeping

- At 1080-point (horizontla), 4 sample per point, BFD 34.25, the matrix took 347.59 seconds to create, and the propagation took 11360.94 seconds. 