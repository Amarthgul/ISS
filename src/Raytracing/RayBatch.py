



from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import NORMAL_RADIANT, INIT_ELLIPSE_TILT, ZERO, ONE, TWO, LambdaLines, RADIANT_KILL, Axis, RNG
from Util.Misc import Normalized


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
    # Φ:                Polarized Radiance term 1   (7)
    # i_Φ:              Polarized Radiance term 2   (8)
    # b:                Polarization ellipse tilt   (9)
    # s:                Surface index               (10)

    def __init__(self, value=None):
        self.value = value
    

    def Position(self):
        return bd.copy(self.value[:, :3])
    

    def Direction(self):
        return bd.copy(self.value[:, 3:6])
    

    def Wavelength(self, singleValue = False):
        """
        Wavelength in nanometer 

        :param singleValue: When enabled, return only the first wavelength of the raybatch.
        """
        if (singleValue):
            return self.value[:, 6][0]
        else:
            return bd.copy(self.value[:, 6])
    

    def Radiannce(self):
        """
        DO NOT USE
        """
        # Technically this is not really radiance since it does not integral over a solid angle 
        return self.value[:, 7]
    

    def PolarizedRadiance(self, polarized=True):
        """
        Radiance as area of the polarization ellipse. 

        :param polarized: defaults to true and will calculate the radiance based on the polarizaton ellipse. Disable this could signifacntly increase memory and speed. 
        """

        # For generalized purpose, this method is used for almost all radiance inquiries. However, calculating the polarized radiance brings a huge memory and speed loss, as such, an additional option is coded here to directly skip the ellipse calculation. 
        if(not polarized):
            return self.Radiannce()

        # Accquire eigen value and eigen vector 
        if(backend_name == "cupy"):
            pol = self.PolarizationMat()
            val, vec = bd.linalg.eigh(pol)
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


    def GetRaysAt(self, index, asRB=True):
        """
        Accquire the rays whose index is at the given value. 

        :param index: surface index. 
        :param asRB: when enabled, return as a RayBatch object.

        :return: array of rays or RayBatch object.
        """

        val = self.value[self.value[:, 10] == index]

        if(asRB):
            return RayBatch(val)
        
        return val


    def GetRaysFacing(self, facingPosZ=True):
        """
        Select the rays that are facing the positive or negative z direction. 
        """
        mask = self.Direction()[:, Axis.Z.value] < 0

        if(facingPosZ):
            mask = ~mask

        return RayBatch(self.value[mask])
            

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
            self.value = input.value
        else:
            self.value = bd.vstack((self.value, input.value))

        return self


    def RandomDrop(self, dropRate = 0.0):
        """Randomly remove some of the rays in this raybatch."""
        rnd = RNG.rand(len(self.value))
        keep_mask = (rnd >= dropRate)

        self.value = self.value[keep_mask]


    def Mask(self, validMask):
        """
        Mask and remove rest of the entries. 

        :param validMask: true for entries that should be kept.

        :return: this RayBatch object itself, which has been masked. 
        """
        self.value = self.value[validMask]
        return self 


    def RadiantKill(self):
        """
        Remove all the rays whose polarized radiance is below the threshold.

        :return: this RayBatch object itself, which has been modidified.
        """
        
        validMask = self.PolarizedRadiance() >= RADIANT_KILL
        self.Mask(validMask)

        return self


    def SurfaceKill(self, index):

        removeMask = self.SurfaceIndex() == index
        self.Mask(~removeMask)


    def TrimExitRays(self, index):

        validMask = (self.value[:, 10] == index) & (self.value[:, 5] > 0)

        exitRB = RayBatch(bd.copy(self.value[validMask]))
        self.Mask(~validMask)

        return exitRB



    def ToString(self, maxCount=50):

        result = "[\n"
        for row in self.value[:maxCount]:
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
    temp[3] = INIT_ELLIPSE_TILT   # Phase difference 
    # 4th term in temp is the durface index 
    
    return RayBatch(
        bd.concatenate([pos, bd.tile(temp, (size, 1))], axis=1)
    ) 


def GenerateBeam(position, direction, size=16, wavelength=LambdaLines['D']):
    """
    Generate a light beam, i.e., a group of rays with the same position and direction.
    """
    direction = Normalized(direction)
    pos = bd.concatenate([position, direction], axis=0)
    pos = bd.tile(pos, (size, 1))
    temp = bd.zeros(5)

    temp[0] = wavelength
    temp[1] = NORMAL_RADIANT    # Sagittal radiant
    temp[2] = NORMAL_RADIANT    # Tangential radiant
    temp[3] = INIT_ELLIPSE_TILT   # Phase difference 
    # 4th term in temp is the durface index 
    
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
