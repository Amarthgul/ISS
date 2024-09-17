

import numpy as np 

class RayBatch:
    """
    Raybatch data are organized in the form of:
    [[x, y, z, v_x, v_y, v_z, lambda, phi, i, bs], [...], ...]
    """
    def __init__(self, value):
        self.value = value

    def Position(self):
        return self.value[:, :3]
    
    def Direction(self):
        return self.value[:, 3:6]
    
    def Wavelength(self):
        """
        Wavelength in nanometer 
        """
        return self.value[:, 6]
    
    def Radiant(self):
        """
        Radiant flux or unitless light intensity 
        """
        return self.value[:, 7]
    
    def SurfaceIndex(self):
        """
        Last surface this ray passed. 
        """
        return self.value[:, 8]
    
    def Sequential(self):
        """
        Whether or not the rays are sequential 
        """
        return self.value[:, 9].astype(bool)

    def SetPosition(self, positions, Sequential=True):
        if(positions.shape[1] != 3): 
            raise ValueError("Expect positions to have a dimension of (#, 3)")
        self.value[np.where(self.value[:, 9] == 1), :3] = positions[:, :]

    def SetDirection(self, directions):
        if(directions.shape[1] != 3): 
            raise ValueError("Expect directions to have a dimension of (#, 3)")
        self.value[:, 3:6] = directions[:, :]

    def SetVignette(self, vignettedIndices):
        self.value[vignettedIndices, 9] = 0