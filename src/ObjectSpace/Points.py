

import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Misc import RGBToWavelengthSameD, Lumi, GridNormalized
from Util.PltPlot import DrawDirection
from Util.Globals import ONE, INIT_PHASE_DIFF
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
            xPos = self.value[:, 0] if self.angleInRad else bd.deg2rad(self.value[:, 0])
            yPos = self.value[:, 1] if self.angleInRad else bd.deg2rad(self.value[:, 1])
            xPos = self.value[:, 2] * bd.tan(xPos)
            yPos = self.value[:, 2] * bd.tan(yPos)
            return bd.column_stack((xPos, yPos, self.value[:, 2]))
            
    
    def Color(self):
        return self.value[:, 3:]


    def EmitTowards(self, targets, jitter=None):
        """
        Emit rays from the point sources towards the target points. 

        :param targets: collection of points as emission targets. 

        :return: raybatch object of rays from the point sources to the target, with corresponding wavelengths. 
        """
        return self._SamplesToTargetsEmission(self, targets)


    def EmitSamplesToward(self, targets, sampleCount=64, jitter=None):
        """
        Emit rays from some of the point sources towards the target points. 

        :param targets: collection of points as emission targets. 
        :param sampleCount: amount of point sources to be sampled from all the soruces.

        :return: raybatch object of rays from the point sources to the target, with corresponding wavelengths. 
        """

        # In the case that there is less sources than sample count, return all 
        if(self.sampleRecord.shape[0] == sampleCount):
            return self.EmitTowards(targets)

        # TODO: add something here to make those with less record selected more prone to be selected 

        # Create a selection array 
        selectedIndices = bd.random.choice(self.value.shape[0], sampleCount, replace=False)

        # Increase the sample records of the selected 
        self.sampleRecord[selectedIndices] += 1

        return self._SamplesToTargetsEmission(
            PointsSource(self.value[selectedIndices]), 
            targets, 
            jitter)
        

    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetSampleRecord(self):
        if(self.value is None):
            return  
        else: 
            self.sampleRecord = bd.zeros(self.value.shape[0]).astype(bd.int32)


    def _SamplesToTargetsEmission(self, sampleSource, targets, jitter=None):
        sourcePos = sampleSource.Position()

        # Expand the points to prepare crossing them 
        sourceExpanded = sourcePos[:, bd.newaxis, :]  # Shape (n, 1, 3)
        targetsExpanded = targets[bd.newaxis, :, :]  # Shape (1, m, 3)

        # Compute the direction of acrossing the source and target 
        dirCross = targetsExpanded - sourceExpanded # Shape (n, m, 3)
        dirCross = GridNormalized(dirCross)

        # Expand and append the position into pos/dir pairs 
        sourcePos = sourcePos[:, bd.newaxis, :]
        # Introduce jitter to the position if needed 
        sourcePos = self._AddJitter(
            bd.tile(sourcePos, (1, dirCross.shape[1], 1)), 
            jitter
            )

        appended = bd.concatenate([sourcePos, dirCross], axis=2)
        # After applying the mask, appended is of shape (m*n', 6)

        # Convert source color to wavelength 
        wavelengths, radiants = RGBToWavelengthSameD(sampleSource.Color())


        # Expand the wavelength to match the pos/dir 
        wavelengths = wavelengths[:, bd.newaxis, :]
        wavelengths = bd.tile(wavelengths, (1, dirCross.shape[1], 1))
        # Spilt the wavelengths, copy and concatenate them to 
        wavelengths = bd.split(wavelengths, indices_or_sections=3, axis=2)

        appended = [bd.concatenate([appended, b], axis=2) for b in wavelengths]
        
        # This creates a boolean mask whose filter ratio is based on the radiant of the corresponding wavelength 
        radiantMask = [
            bd.random.random((sourcePos.shape[0], targets.shape[0])) < radiants[:, i][:, bd.newaxis] 
            for i in range(radiants.shape[1])
            ] 

        # Using the filter from last step to drop the elements.
        # This is how RGB is created, note that such method rely heavily on large scale Monte Carlo to reduce randomness 
        appended = [appended[i][radiantMask[i]] for i in range(len(radiantMask))]


        # This yields a (w*m*n', 7) array, prime sign means it's smaller than w*m*n since some of them are just dropped out 
        appended = bd.concatenate(appended, axis=0) 
        
        temp = bd.ones(3)
        temp[0] = ONE    # Sagittal radiant
        temp[1] = ONE    # Tangential radiant
        temp[2] = INIT_PHASE_DIFF   # Phase difference 

        return RayBatch(
            bd.concatenate([appended, bd.tile(temp, (appended.shape[0], 1))], axis=1)
        )


    def _AddJitter(self, input, jitterAmount):
        """
        This method adds a jittering onto the input, with the range limited to a certain amount. When applied on point positions, this makes them more like sampling a continuous signal rather than a bunch of idealized discrete points in space. This also has anti-aliasing effect in many occasions. 

        :param input: array of whatever size. 
        :param jitterAmount: scalar of abs value range of the jitter efect. 

        :return: a new array of same size as input but with value jittered. 
        """

        if(jitterAmount is None):
            return input
        
        # randArr = bd.random.uniform(low=-0.5, high=0.5, size=input.shape)
        # print(bd.mean(randArr))
        # print(bd.std(randArr))
        return input + jitterAmount * bd.random.uniform(low=-0.5, high=0.5, size=input.shape)




def main():
    
    t = PointsSource(bd.array([
        #[0,  0, -10000, 1, 1, 1], 
        [0, 10, -10000, 1, 0.6, 0.6],
        [0, 25, -10000, 1, 0.5, 0.2],
    ]))
    t.isCartesian = False

    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    RB = t.EmitTowards(targets)

    print(RB.value)



if __name__ == "__main__":
    main() 

