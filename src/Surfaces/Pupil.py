

import warnings


from Util.Globals import ZERO, RNG
from Util.Misc import Normalized, ArrayMagnitude, ColorTuplePLT, WavelengthToRGB, MovingAverageSmoothing, GaussianSmooth
from Util.Backend import backend as bd 
from Util.Sampling import RandomEllipticalDistribution, PoissonDiskDistribution
from Util.PltPlot import DrawDisk, DrawPupil, DrawPoints


from .VirtualSurface import VirtualSurface, SymmetryType


class Pupil(VirtualSurface):
    def __init__(self):
        
        self.symmetryType = SymmetryType.Axial
        # By default the pupil is axial symmetric 

        self.clearSemiDiameter = None 

        """The wavelength for the sample points"""
        self.sampleWavelength = None 

        # This set of points represent the pupil size theoretically possible 
        self._height = []  # Transversal distance from the optical axis
        self._zDepth = []  # Axial distance from the origin 

        # The maximum possible pupil size, derived from the theoretical height 
        self._maxPupilSD = None

        # This set of points represent the pupil size that is currently used at the current f-number
        self._workingHeight = []
        self._workingDepth = []

        # The sample pool of points for the pupil at current size 
        self._pupilPointSamples = None


    def AddSamplePoint(self, point):
        """
        Add a single point into the pupil samples.
        """
        self._zDepth.append(point[2])
        self._height.append(Normalized(bd.array([point[0], point[1]])))

        # The newly inserted value may cause trouble when plotting 
        # A re-sorting is needed to ensure both array are in order 
        sortedIndices = bd.argsort(self._height)
        self._zDepth = self._zDepth[sortedIndices]
        self._height = self._height[sortedIndices]


    def SetSamplePoints(self, points):
        """
        Set the sample points, this will override all previous points. Note that this also assumes the points given represents the largest possible pupil size. 
        """
        self._zDepth = points[:, 2]
        self._height = bd.linalg.norm(points[:, :2], axis=1)

        self._maxPupilSD = bd.max(self._height)

        # Copy the theoretical pupil size to the working size
        self._workingDepth = self._zDepth.copy()
        self._workingHeight = self._height.copy()

        self._ResetSamplePool()


    def DrawSurface(self, overrideColor=None):
        """
        Draw the pupil surface at the given aperture (f-number). 

        :param overrideColor: color to override the default blue color.
        """
        wlColor = 'b'

        if (not self.sampleWavelength == None):
            wlColor = ColorTuplePLT(WavelengthToRGB(self.sampleWavelength))
        if(not overrideColor == None):
            wlColor = overrideColor

        # When there is only one data point,
        if(len(self._zDepth) == 1):
            # Assume it is the center point on axis and use it as the overall depth 
            DrawDisk(self.clearSemiDiameter, self._zDepth[0], surfaceColor=wlColor)

        # When there are many different points for the pupil plane 
        else:
            DrawPupil(self._workingHeight, self._workingDepth, surfaceColor=wlColor)


    def DrawSamplePoints(self, overrideColor=None, duplicateAxial=True, smoothPoints=True):
        """
        Draw the sample poinst that are used to define the pupil.

        :param overrideColor: color to override the default blue color.
        :param duplicateAxial: duplicate the points on the opposite side of the axis.
        :param smoothPoints: smooth the points using a Gaussian filter.
        """

        wlColor = 'b'

        if (not self.sampleWavelength == None):
            wlColor = ColorTuplePLT(WavelengthToRGB(self.sampleWavelength))
        if(not overrideColor == None):
            wlColor = overrideColor

        smoothDepth = self._zDepth
        if(smoothPoints):
            smoothDepth = GaussianSmooth(self._zDepth)

        points = bd.stack(
            (bd.zeros(len(self._height)), self._height, smoothDepth), axis=-1)

        if(duplicateAxial):
            opposite = bd.stack(
                (bd.zeros(len(self._height)), -self._height, smoothDepth), axis=-1)
            points = bd.vstack((points, opposite))


        DrawPoints(points, color=wlColor)


    def SetPupilSize(self, semiDiameter):
        """
        Set the clear semi-diameter of the pupil, also regenerates the sample pool. This can be used to reflect the change of f-number.
        """

        # Check if the semi-diameter given is plausible
        if(semiDiameter > self._maxPupilSD):
            warnings.warn("The pupil size is larger than possible. Max possible size is used instead.")
            semiDiameter = self._maxPupilSD
        
        self.clearSemiDiameter = semiDiameter

        self._ResetWorkingPupilSize()
        self._ResetSamplePool()


    def GetMaxPupilSize(self):
        """
        Get the maximum possible pupil size.
        """
        return self._maxPupilSD * 2


    def GetSamplePoints(self, sampleCount):
        """
        Get some sample points that are on the pupil. 
        """
        selectedIndices = bd.random.choice(self._pupilPointSamples.shape[0], sampleCount, replace=False)

        return self._pupilPointSamples[selectedIndices]


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _ResetWorkingPupilSize(self):
        """
        Reset the working pupil size to the current clear semi-diameter.
        """

        # Find the index where the clear semi-diameter should be inserted
        insert_index = bd.searchsorted(self._height, self.clearSemiDiameter)

        # Compute the interpolated value for axial depth 
        self._workingDepth = bd.interp(self.clearSemiDiameter, self._height, self._zDepth)

        # Insert the clear semi-diameter and the interpolated depth and assign them to the working pupil points 
        self._workingHeight = bd.concatenate([self._height[:insert_index], bd.array([self.clearSemiDiameter])])
        self._workingDepth = bd.concatenate([self._zDepth[:insert_index], bd.array([self._workingDepth])])
        

    def _ResetSamplePool(self, poolSize=128):
        """
        Regenerate the sample pool of points for the pupil based on the current pupil shape and size.
        """

        # Using the curved pupil surface might introduce some unevenness after propagation, an average is used here. Other methods can also be used to calculate the average depth depending on the desired effect.
        pupilZdepth = bd.mean(self._workingDepth)
        # Using the working depth instead of the theoretical depth so that focus shift caused by different f-stop can be somewhat simulated.


        # TODO: Currently this assumes the shape of the pupil is circular, in the future, other shapes can be added.
        self._pupilPointSamples = RandomEllipticalDistribution(
            major_axis=self.clearSemiDiameter,
            minor_axis=self.clearSemiDiameter,
            samplePoints=poolSize, 
            zDepth=pupilZdepth, 
            groupByPoint=True)
        
        #self._pupilPointSamples = PoissonDiskDistribution()

        
