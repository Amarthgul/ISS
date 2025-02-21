

from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import NORMAL_RADIANT, INIT_PHASE_DIFF, ZERO, ONE, TWO, LambdaLines


class RayBatch:
    """
    Raybatch data are organized in the form of:
    [
       [x, y, z, v_x, v_y, v_z, λ, Φ, i_Φ, b, s], 
       [...], [...], ...
    ]
    """
    # x, y, z:          Root position of the ray    (0, 1, 2)
    # v_x, v_y, v_z:    Direction of the ray        (3, 4, 5)
    # λ:                Wavelength of the ray       (6)
    # Φ:                Polarization term 1         (7)
    # i_Φ:              Polarization term 2         (8)
    # b:                Polarization ellipse tilt   (9)
    # s:                Surface index               (10)

    def __init__(self, value=None):
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
        # TODO: consider adding the polarization direction here? 
        return self.value[:, 7]
    

    def PolarizedRadiance(self):
        """
        Radiance as area of the polarization ellipse. 
        """

        # Accquire eigen value and eigen vector 
        if(backend_name == "cupy"):
            val, vec = bd.linalg.eigh(self.PolarizationMat())
        else:
            val, vec = bd.linalg.eig(self.PolarizationMat())

        # Semi axis of the polariztion ellipse 
        semi = ONE / bd.sqrt(val)

        # To ensure radiance conservation, a simple addition and normalization is used here
        return (semi[:, 0] + semi[:, 1]) / 2


    def PolarizationMat(self):
        """
        Return the radiance and polarization term as an ellipse matrix. 
        """
        sliced = self.value[:, [7, 9, 8]]  

        part1 = sliced[:, :2]  
        part2 = sliced[:, 1:]  

        return bd.stack([part1, part2], axis=1)
    

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


    def SetIndex(self, indices):

        self.value[:, 10] = indices


    def SetPolarization(self, polarM):
        """
        Given the polarization matrices, decompose them and assign them to the raybatch.
        """
        a = polarM[:, 0, 0]
        c = polarM[:, 0, 1] # The tilt term 
        b = polarM[:, 1, 1]

        temp = bd.stack((a, b, c), axis=0).T

        self.value[:, 7:10] = temp


    def SetPolarizationPerTerm(self, diag1=None, diag2=None, tilt=None):
        """
        Force override the polarization terms. 
        """

        if(diag1 is not None):
            self.value[:, 7] = diag1
        if(diag2 is not None):
            self.value[:, 8] = diag2
        if(tilt is not None):
            self.value[:, 9] = tilt


    def Merge(self, input):
        """
        Merge this raybatch with another one. This raybatch will be modifed and also as a return value.
        """

        if(self.value is None):
            self.value = input
        else:
            self.value = bd.vstack((self.value, input.value))

        return self


    def RandomDrop(self, keeprate = 1):
        pass 


    def Mask(self, mask):
        """
        Mask and remove exposed entries. 
        """
        self.value = self.value[mask]
        return self 


    def ToString(self):
        result = "[\n"
        for row in self.value:
            # Convert each row's elements to strings and join them with commas
            row_str = "  [" + ", ".join(str(x) for x in row) + "],\n"
            result += row_str
        
        result += "]"
        return result



def GenerateEmpty(size=16, wavelength=LambdaLines['D']):

    pos = bd.zeros(6)
    pos = bd.tile(pos, (size, 1))
    temp = bd.zeros(5)
    temp[0] = wavelength
    temp[1] = NORMAL_RADIANT    # Sagittal radiant
    temp[2] = NORMAL_RADIANT    # Tangential radiant
    temp[3] = INIT_PHASE_DIFF   # Phase difference 
    
    return RayBatch(
        bd.concatenate([pos, bd.tile(temp, (size, 1))], axis=1)
    ) 


def main():
    A = RayBatch(bd.array([
        [0, 0, 0, 0, 0, 1, 550, 1, 1, 0, 0], 
        [0, 0, 0, .1, .1, .9, 550, 1.50 , 4.18, 1.27, 0]]))
    B = RayBatch(bd.array([
        [1, 0, 0, 0, 0, 1, 550, 1, 1, 0, 0], 
        [0, 1, 0, .1, .1, .9, 550, 1.50 , 4.18, 1.27, 0]]))
    
    #A.Merge(B)
    A.SetPolarization(
        bd.array([[[1, 0], [0, 1]], 
                  [[2, .5], [.5, 1]]]))

    print(A.ToString())


if __name__ == "__main__":
    main()
