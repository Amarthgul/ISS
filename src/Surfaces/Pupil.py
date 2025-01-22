

import warnings


from Util.Globals import ZERO, RNG
from Util.Misc import Normalized, ArrayMagnitude, ColorTuplePLT, WavelengthToRGB, MovingAverageSmoothing, GaussianSmooth, RandomEllipticalDistribution
from Util.Backend import backend as bd 
from Util.PltPlot import DrawDisk, DrawPupil, DrawPoints


from .VirtualSurface import VirtualSurface, SymmetryType


class Pupil(VirtualSurface):
    def __init__(self):
        
        self.symmetryType = SymmetryType.Axial
        # By default the pupil is axial symmetric 

        self.clearSemiDiameter = None 

        """The wavelength for the sample points"""
        self.sampleWavelength = None 

        self._height = []
        self._zDepth = []


        self._maxPupilSize = None


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
        Set the sample points, this will override all previous points. 
        """
        self._zDepth = points[:, 2]
        self._height = bd.linalg.norm(points[:, :2], axis=1)

        self._maxPupilSize = bd.max(self._height)


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
            heightConstraint = self._height < self.clearSemiDiameter
            newHeight = self._height[heightConstraint]
            newDepth = self._zDepth[heightConstraint]
            newHeight[bd.argmax(newHeight)] = self.clearSemiDiameter

            DrawPupil(newHeight, newDepth, surfaceColor=wlColor)


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
        if(semiDiameter > self._maxPupilSize):
            warnings.warn("The pupil size is larger than possible. Max possible size is used instead.")
            semiDiameter = self._maxPupilSize
        
        self.clearSemiDiameter = semiDiameter

        self._ResetSamplePool()


    def GetSamplePoints(self, sampleCount):
        """
        Get some sample points that are on the pupil. 
        """
        selectedIndices = bd.random.choice(self._pupilPointSamples.shape[0], sampleCount, replace=False)

        return self._pupilPointSamples[selectedIndices]


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetSamplePool(self, poolSize=128):
        """
        Regenerate the sample pool of points for the pupil. 
        """

        # Using the curved pupil surface might introduce some unevenness after propagation, an average is used here. Other methods can also be used to calculate the average depth depending on the desired effect.
        pupilZdepth = bd.mean(self._zDepth)


        # TODO: Currently this assumes the shape of the pupil is circular, in the future, other shapes can be added.
        self._pupilPointSamples = RandomEllipticalDistribution(
            major_axis=self.clearSemiDiameter,
            minor_axis=self.clearSemiDiameter,
            samplePoints=poolSize, 
            zDepth=pupilZdepth).T

        print("Somehing")
        
