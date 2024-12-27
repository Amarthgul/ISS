

from Util.Backend import backend as bd 

class RayBatch:
    """
    Raybatch data are organized in the form of:
    [
       [x, y, z, v_x, v_y, v_z, λ, Φ, i_Φ, pd, s], 
       [...], [...], ...
    ]
    """
    # x, y, z:          Root position of the ray    (0, 1, 2)
    # v_x, v_y, v_z:    Direction of the ray        (3, 4, 5)
    # λ:                Wavelength of the ray       (6)
    # Φ:                Radiant (Sagittal)          (7)
    # i_Φ:              Radiant (Rangential)        (8)
    # pd:               Phase difference            (9)
    # s:                Surface index               (10)

    def __init__(self, value):
        self.value = value
    

    def Position(self):
        return self.value[:, :3]
    

    def Direction(self):
        return self.value[:, 3:6]
    

    def Wavelength(self, singleValue = False):
        """
        Wavelength in nanometer 

        :param singleValue: When enabled, return only the first wavelength of the raybatch.
        """
        if (singleValue):
            return self.value[:, 6][0]
        else:
            return self.value[:, 6]
    

    def Radiant(self):
        """
        Radiant flux or unitless light intensity 
        """
        return self.value[:, 7]
    

    def RadiantImaginary(self):
        """
        Radiant flux or unitless light intensity 
        """
        return self.value[:, 8]


    def PhaseDifference(self):
        """
        Radiant flux or unitless light intensity 
        """
        return self.value[:, 9]
    

    def SurfaceIndex(self):
        """
        Last surface this ray passed. 
        """
        return self.value[:, 10]


    def SetPosition(self, positions):
        if(positions.shape[1] != 3): 
            raise ValueError("Expect positions to have a dimension of (#, 3)")
        self.value[:, :3] = positions[:, :]


    def SetDirection(self, directions):
        if(directions.shape[1] != 3): 
            raise ValueError("Expect directions to have a dimension of (#, 3)")
        self.value[:, 3:6] = directions



    def RandomDrop(self, keeprate = 1):
        pass 




