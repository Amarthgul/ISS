



from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import NORMAL_RADIANT, INIT_ELLIPSE_TILT, ZERO, ONE, TWO, LambdaLines, RADIANT_KILL, Axis, RNG
from Util.Misc import Normalized


class RayBatch:
    """
    Raybatch data are organized in the form of:
    [
       [x, y, z, v_x, v_y, v_z, λ, Φ, i_Φ, b, s, C, (optional)AOV1, (optional)AOV2, ...],
       [...], [...], ...
    ]
    """
    # x, y, z:          Root position of the ray    (0, 1, 2)
    # v_x, v_y, v_z:    Direction of the ray        (3, 4, 5)
    # λ:                Wavelength of the ray in nm (6)
    # Φ:                Polarized Radiance term 1   (7)
    # i_Φ:              Polarized Radiance term 2   (8)
    # b:                Polarization ellipse tilt   (9)
    # s:                Surface index               (10)
    # C:                Color channel of RGB        (11)
    # Optional AoVs                                 (12+)

    def __init__(self, value=None):
        self.value = value
    

    def Position(self):
        return bd.copy(self.value[:, :3])
    

    def Direction(self):
        return bd.copy(self.value[:, 3:6])
    

    def Transform(self, transformationMatrix):
        pass


    def Wavelength(self, singleValue = False):
        """
        Wavelength in nanometer 

        :param singleValue: When enabled, return only the first wavelength of the raybatch.
        """
        if (singleValue):
            return self.value[:, 6][0]
        else:
            return bd.copy(self.value[:, 6])
    

    def RadianceTerms(self):
        """
        Get the 3 constituents of the polarized radiance.
        """
        return self.value[:, [7, 9, 8]]


    def Radiance(self):
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
            return self.Radiance()

        # Accquire eigen value and eigen vector 
        # if(backend_name == "cupy"):
        #     pol = self.PolarizationMat()
        #     val, vec = bd.linalg.eigh(pol)
        # else:
        #     val, vec = bd.linalg.eig(self.PolarizationMat())

        pol = self.PolarizationMat()
        # Use Hermitian eigensolver for symmetric 2x2; gives real, ordered eigenvalues
        val, _ = bd.linalg.eigh(pol)
        # Clamp to avoid negatives/zeros due to numerical noise
        eps = 1e-12
        val = bd.maximum(val, eps)


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


    def SanitizePolarization(self):
        """
        Drop rays whose polarization parameters (columns 7,8,9) are missing or non-finite.
        Returns (self, removed_count).
        """

        rad = self.PolarizedRadiance()

        noneMask = bd.array([x is None for x in rad.get()])
        good = rad >= 0 & rad <= 1
        good = good & ~noneMask

        self.value = self.value[good]

        return self


    def SurfaceIndex(self):
        """
        Last surface this ray passed. 
        """
        return self.value[:, 10]


    def Channel(self):
        return  self.value[:, 11]


    def GetRaysAt(self, index, asRB=True):
        """
        Acquire the rays whose index is at the given value.

        :param index: surface index. 
        :param asRB: when enabled, return as a RayBatch object.

        :return: array of rays or RayBatch object.
        """

        val = self.value[self.value[:, 10] == index]

        if(asRB):
            return RayBatch(val)
        
        return val


    def GetDirectionalRay(self, facingPosZ=True):
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
        """
        Set the surface index of this raybatch.
        """
        self.value[:, 10] = indices


    def SetPolarization(self, polarM):
        """
        Given the polarization matrices, decompose them and assign them to the raybatch.

        :param polarM: polarization matrices, each entry is a 2x2 matrix.
        """
        a = polarM[:, 0, 0]
        c = polarM[:, 0, 1] # The tilt term 
        b = polarM[:, 1, 1]

        temp = bd.stack((a, b, c), axis=0).T

        self.value[:, 7:10] = temp


    def SetRadianceTerms(self, radianceTerms):
        """
        Replace the radiance terms with the given ones.

        :param radianceTerms: list of radiance terms, each as an array of shape (3).
        """

        self.value[:, [7, 9, 8]] = radianceTerms


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


    def GetAoV(self):
        return self.value[:, 11:]


    def AppendAOV(self, values):
        """
        Append an AOV into this raybatch.
        :param values: an array of AOV values whose first dimension must be the same as the value of the current raybatch.

        :return: this RayBatch object.
        """
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        # Directly concatenate along the feature / column axis
        self.value = bd.concatenate((self.value, values), axis=1)

        return self


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


    def RadiantKill(self, killThreshold = RADIANT_KILL):
        """
        Remove all the rays whose polarized radiance is below the threshold.

        :param killThreshold: threshold for ray kill.
        :return: this RayBatch object itself, which has been modified.
        """
        
        validMask = self.PolarizedRadiance() >= killThreshold
        self.Mask(validMask)

        return self


    def SurfaceKill(self, index):
        """
        Kill all the rays at a surface.

        :param index: surface index at which the rays shall be killed.
        """
        removeMask = self.SurfaceIndex() == index
        self.Mask(~removeMask)


    def TrimExitRays(self, index):
        """
        Trim out the rays facing at object space at a surface.

        :param index: surface index at which the rays shall be examined and trim.
        """
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


    def SurfaceDistributionInfo(self):
        """
        Return a string summary of how many rays belong to each surface index.
        Format:
            0: m
            1: n
            2: ...
        """

        if self.value is None or self.value.shape[0] == 0:
            return "No rays in batch."

        # Assume surface index is stored in column self.COL_SURFACE
        surf_idx = bd.asarray(self.value[:, 10]).astype(int)

        # Count occurrences of each unique surface index
        unique_surfaces, counts = bd.unique(surf_idx, return_counts=True)

        # Convert to CPU if backend is CuPy
        if hasattr(unique_surfaces, "get"):
            unique_surfaces = unique_surfaces.get()
            counts = counts.get()

        # Build formatted string
        lines = [f"{int(s)}: {int(c)}" for s, c in zip(unique_surfaces, counts)]
        return "\n".join(lines)


    def IsNone(self):

        return self.value is None


    def IsNoneType(self):

        if self.value is not None:
            return len(self.value) == 0
        else:
            return True


    def Copy(self):
        """
        Return a deep of this RayBatch instance.

        :return: A fully independent copy of the current object.
        """

        # Create a new RayBatch instance without sharing memory
        new_rb = RayBatch(bd.copy(self.value))

        return new_rb





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
