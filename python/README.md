

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


# Record Keeping

- At 1080-point (horizontla), 4 sample per point, BFD 34.25, the matrix took 347.59 seconds to create, and the propagation took 11360.94 seconds. 