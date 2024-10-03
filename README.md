# IOSS
Implicit Optical System Simulation 



## Code Conventions


### Underscore 

Single underscore methods means they are private to the class and is required for the class to perform its duty. Double underscore methods are for internal testing purpose only and typically not used when the class is called externally. 

For example, the method `ellipsePeripheral()` appeared both in the `Lens` class and in the `ImagingSystem` class. However, in the `Lens` class it is named `__ellipsePeripheral()`, as it is only used for testing the lens class during development and not a part of the functionality for the `Lens` class by contract.  