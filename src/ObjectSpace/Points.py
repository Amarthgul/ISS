

import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ColorWavelength import RGBToWavelengthSameD, RGBToWavelengthSpotSim, Lumi
from Util.Misc import  GridNormalized, PolarToCartesian
from Util.PltPlot import DrawDirection, DrawPoints, DrawPointsPerColor, DrawRaybatch, SetUnifScale
from Util.Globals import ONE, INIT_ELLIPSE_TILT, FAR_DISTANCE, RNG, RefreshRNG
from Raytracing.RayBatch import RayBatch




class PointsSource:
    """
    Point sources are organized in the form of:
    [[x, y, z, R, G, B], 
       [...], [...], ...]
    RGB must be float number in the range of [0, 1]. 
    Alternatively, it may also be using field angle representation:
    [[θ_x, θ_y, D, R, G, B], 
       [...], [...], ...]] 
    Where D is the distance from the front vertex of the lens, which in this case is the polar origin.
    """

    def __init__(self, data=None):
        self.value = data


        """Whether the data is Cartesian XYZ coordinates or field angles"""    
        self.isCartesian = True
        # This should be set to True by default. The only case polar coordinate is useful is spot testing, which can be manually entered, flagged, and adjusted. 


        """Whether the angle is in radian"""
        self.angleInRad = False


        self.sampleRecord = None 
        self._ResetSampleRecord()


    def SetPoints(self, points):
        self.value = points
        self._ResetSampleRecord()


    def AddPoint(self, point):
        if(self.value is None):
            self.value = point
        else:
            bd.vstack(self.value, point)
        self._ResetSampleRecord()


    def Position(self):
        """
        Calculate and return the Cartesian coordinate of the points. If the point sources are created using field angles, i.e., polar coordinates, they will be converted to Cartesian.  
        """
        if(self.isCartesian):
            return self.value[:, :3]
        else:
            return PolarToCartesian(self.value[:, :3], True)
            
    
    def Color(self):
        return self.value[:, 3:]


    def EmitSamplesToward(self, targets, sampleCount=64, jitter=None, addSecondary=None):
        """
        Emit rays from some of the point sources towards the target points. 

        :param targets: collection of points as emission targets. 
        :param sampleCount: amount of point sources to be sampled from all the soruces.

        :return: raybatch object of rays from the point sources to the target, with corresponding wavelengths. 
        """

        # In the case that there are fewer sources than demanded sample count, return all
        if(self.sampleRecord.shape[0] <= sampleCount):
            return self._SamplesToTargetsEmission(self, targets, jitter=jitter, addSecondary=addSecondary)

        # Select the ones that have been sampled the least to ensure the sample is relatively even 
        selectedIndices = self._SelectLeastSampled(sampleCount)

        # Increase the sample records of the selected 
        self.sampleRecord[selectedIndices] += 1

        # Create a new source instance using the selected points 
        sourceDuplicate = PointsSource(self.value[selectedIndices])

        # In case of spot testing, copy the Cartesian setting 
        sourceDuplicate.isCartesian = self.isCartesian

        # If the sources come from a vari depth image, the jitters may be a 1D array corresponding to each source
        if (bd.ndim(jitter) == 1): jitter = jitter[selectedIndices]

        return self._SamplesToTargetsEmission(
            sourceDuplicate, 
            targets, 
            jitter, 
            addSecondary)
        

    def GenerateSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=5):
        """
        Generate white point sources convering from the axis to the postive direction of the given x and y angle. 
        """

        xAngles = bd.linspace(0, xAngle, sampleField)
        yAngles = bd.linspace(0, yAngle, sampleField)

        dists = - bd.ones_like(xAngles) * bd.array(dist)

        #append = self._PolarToCart(bd.column_stack((xAngles, yAngles, dists)))
        append = bd.column_stack((xAngles, yAngles, dists))

        # Concatenate the two arrays along axis 1 (columns)
        self.value = bd.concatenate([append, bd.full((append.shape[0], 3), 1)], axis=1)

        self._ResetSampleRecord()


    def GenerateGridSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=10):
        """
        Generate a symmetric 2D grid of white point sources spanning from 
        (-xAngle, -yAngle) to (xAngle, yAngle).
        """

        # Create symmetric angle ranges for both axes
        xAngles = bd.linspace(-xAngle, xAngle, sampleField)
        yAngles = bd.linspace(-yAngle, yAngle, sampleField)

        # Create 2D grid of (x, y) angles
        xGrid, yGrid = bd.meshgrid(xAngles, yAngles)

        # Flatten the mesh to 1D arrays
        xFlat = xGrid.ravel()
        yFlat = yGrid.ravel()

        # All distances set to a negative value (e.g., for direction into scene)
        dists = -bd.ones_like(xFlat) * dist

        # Combine into Nx3 coordinates: (xAngle, yAngle, distance)
        gridPoints = bd.column_stack((xFlat, yFlat, dists))

        # Append a white intensity vector (1, 1, 1) to each point
        fullPoints = bd.concatenate([gridPoints, bd.full((gridPoints.shape[0], 3), 1)], axis=1)

        # Store the full array
        self.value = fullPoints

        # Reset sample record
        self._ResetSampleRecord()


    def GenerateFixPoint(self, position=None, color=bd.array([1, 1, 1])):

        if(position is None):
            position = bd.array([0, 0, FAR_DISTANCE])

        assemb = bd.concatenate([position, color])
        self.value = bd.stack([assemb, assemb], axis=0)
        self._ResetSampleRecord()
        

    def GetSampleRatios(self):
        """
        This method examines the current sample record and returns the min, the max, and the ratio between min and max. The return van be used as a level of completion estimate or a sample space signal to noise ratio examination. 
        """
        minSampleCount = bd.min(self.sampleRecord)
        maxSampleCount = bd.max(self.sampleRecord)

        return minSampleCount, \
                maxSampleCount, \
                minSampleCount/maxSampleCount


    def DrawPoints(self):
        """
        Draw the points sources in 3D space with corresponding colors.
        """
        DrawPointsPerColor(self.Position(), self.Color())


    def ToString(self):
        result = "[\n"
        for row in self.value:
            # Convert each row's elements to strings and join them with commas
            row_str = "  [" + ", ".join(str(x) for x in row) + "],\n"
            result += row_str
        
        result += "]"
        return result


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetSampleRecord(self):
        if(self.value is None):
            return  
        else: 
            self.sampleRecord = bd.zeros(self.value.shape[0]).astype(bd.int32)


    def _SamplesToTargetsEmission(self, sampleSource, targets, jitter=None, addSecondary=None, cosineFalloff=True):
        """
        Emit rays from all samples towards all targets. This creates a cross emission of MxN size where M is the number of samples and N is the number of targets.

        :param sampleSource: sample source as point source objects.
        :param targets: targets of 3D positions.
        :param jitter: jittering amount, either a single float or a vector of floats corresponding to all the sample sources.
        :param addSecondary: whether to add secondary spectrum into the wavelength or not.
        :param cosineFalloff: whether to consider cosine 4th power falloff.

        :return: RayBatch object of the emitted rays.
        """

        sourcePos = bd.copy(sampleSource.Position())
        sourcePos = sourcePos[:, bd.newaxis, :]
        sourcePos = bd.tile(sourcePos, (1, targets.shape[0], 1))
        if(jitter is not None):
            RefreshRNG()
            if bd.ndim(jitter) == 0:  # scalar → broadcast
                mag = bd.full(sourcePos.shape[:2], float(jitter))

            elif bd.ndim(jitter) == 1:  # 1-D, per source
                if jitter.shape[0] != sourcePos.shape[0]:
                    raise ValueError("jitter 1-D length must equal number of sources")
                mag = bd.tile(jitter[:, bd.newaxis], (1, targets.shape[0]))

            mag_xyz = mag[..., bd.newaxis] * bd.array([1, 1, 0])
            jitter_off = RNG.uniform(low=-mag_xyz, high=mag_xyz)
            sourcePos += jitter_off

            # jitter = RNG.uniform(-jitter, jitter, (sourcePos.shape[0], targets.shape[0], 3)) * bd.array([1, 1, 0])
            # sourcePos += jitter

        # Expand the points to prepare crossing them
        targetsExpanded = targets[bd.newaxis, :, :]  # Shape (1, m, 3)

        # Compute the direction of acrossing the source and target
        dirCross = targetsExpanded - sourcePos # Shape (n, m, 3)
        dirCross = GridNormalized(dirCross)


        appended = bd.concatenate([sourcePos, dirCross], axis=2)
        # After applying the mask, appended is of shape (m*n', 6)

        # Convert source color to wavelength
        if (addSecondary is not None):
            # Spot sim assumes source to be white, so add more spectrums
            wavelengths, radiants = RGBToWavelengthSpotSim(sampleSource.Color(), addCount=addSecondary)
        else:
            wavelengths, radiants = RGBToWavelengthSameD(sampleSource.Color())


        # Expand the wavelength to match the pos/dir
        wavelengths = wavelengths[:, bd.newaxis, :]
        wavelengths = bd.tile(wavelengths, (1, dirCross.shape[1], 1))
        # At this point the wavelengths should be of size (m, n, n_lambda)
        # Where m is number of source point, n is number of target points
        # and n_lambda is number of different wavelengths

        # Accquire the number of wavelengths,
        wavelengthCount = wavelengths.shape[2]

        # Spilt the wavelengths, copy and concatenate them to
        wavelengths = bd.split(wavelengths, indices_or_sections=wavelengthCount, axis=2)

        appended = [bd.concatenate([appended, b], axis=2) for b in wavelengths]

        # This creates a boolean mask whose filter ratio is based on the radiant of the corresponding wavelength
        radiantMask = [
            bd.random.random((sourcePos.shape[0], targets.shape[0])) < radiants[:, i][:, bd.newaxis]
            for i in range(radiants.shape[1])
            ]

        if(cosineFalloff):
            # 1. cosθ between each ray and +Z (= |z-component|, dirCross already normalised)
            cosTheta = bd.abs(dirCross[..., 2]) ** 4 # shape (n,m)
            # 2. same-size random array
            randCos = bd.random.random(cosTheta.shape)
            # 3. boolean acceptance by comparing randCos < cosTheta
            angMask = randCos < cosTheta  # shape (n,m) boolean
            # 4. combine with the wavelength radiant mask
            radiantMask = [mask & angMask for mask in radiantMask]


        # Using the filter from last step to drop the elements.
        # This is how RGB is created, note that such method rely heavily on large scale Monte Carlo to reduce randomness
        appended = [appended[i][radiantMask[i]] for i in range(len(radiantMask))]


        # This yields a (w*m*n', 7) array, prime sign means it's smaller than w*m*n since some of them are just dropped out
        appended = bd.concatenate(appended, axis=0)

        temp = bd.ones(4)
        temp[0] = ONE    # Sagittal radiant
        temp[1] = ONE    # Tangential radiant
        temp[2] = INIT_ELLIPSE_TILT   # Phase difference
        temp[3] = bd.zeros_like(temp[3])  # Surface index

        return RayBatch(
            bd.concatenate([appended, bd.tile(temp, (appended.shape[0], 1))], axis=1)
        )

        
    def _SelectLeastSampled(self, sampleCount):
        """
        Find all indices with the least amount of sample history.
        """

        min_count = bd.min(self.sampleRecord)
        candidates = bd.where(self.sampleRecord == min_count)[0]
        
        # Shuffle candidates to break ties randomly
        RNG.shuffle(candidates)
        
        if len(candidates) >= sampleCount:
            return candidates[:sampleCount]
        
        # Get remaining points sorted by sample count
        sorted_indices = bd.argsort(self.sampleRecord)
        remaining_indices = sorted_indices[len(candidates):]
        
        # Shuffle remaining indices before selection
        RNG.shuffle(remaining_indices)
        
        selected = bd.concatenate([candidates, remaining_indices[:sampleCount - len(candidates)]])
        return selected[:sampleCount]
    



def main():
    

    t = PointsSource(bd.array(
        [[1., 1., 0, 1, 1, 1], 
         [-1, 1., 0, 1, 1, 1], 
         [-1, -1, 0, 1, 1, 1], 
         [1., -1, 0, 1, 1, 1]]
    ))
    t = PointsSource()
    t.GenerateFixPoint(bd.array([0, 0, -12]))

    targets = bd.array(
        [[1., 1., 4], 
         [-1, 1., 4], 
         [-1, -1, 4], 
         [1., -1, 4]]
    )
    print(t.Position())
    #DrawPoints(t.Position(), ptSize=1)
    DrawPoints(targets, ptSize=1)

    for i in range(10):
        pos = []
        RB = t.EmitSamplesToward(targets, sampleCount=16, jitter=1)
        DrawRaybatch(RB, arrowRatio=0, lLength=4.75)
        pos.append(RB.Position())
        DrawPoints(RB.Position(), ptSize=4)

    
    SetUnifScale(5)
    plt.show()
    
    



if __name__ == "__main__":
    main() 

