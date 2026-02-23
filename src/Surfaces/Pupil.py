

import warnings

from Util.Globals import ZERO, RNG
from Util.ColorWavelength import ColorTuplePLT, WavelengthToRGB
from Util.Misc import Normalized, ArrayMagnitude, MovingAverageSmoothing, GaussianSmooth
from Util.Backend import backend as bd 
from Util.Sampling import RandomEllipticalDistribution, PoissonDiskDistribution, CircularDistribution
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

        self._firstElementSD = None

        # This set of points represent the pupil size that is currently used at the current f-number
        self._workingHeight = []
        self._workingDepth = []

        # The sample pool of points for the pupil at current size 
        self._pupilPointSamples = None

        self._alphaShape = None


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
        Set the sample points of the pupil shape, this will override all previous points. Note that this also assumes the points given represents the largest possible pupil size. 
        """
        self._zDepth = points[:, 2]
        self._height = bd.linalg.norm(points[:, :2], axis=1)
        self.vertex = points[0]

        # Since this method reset all sample points, the max pupil size is copied from the theoretical maxmimum size, this could be changed later if pupil size is changed. 
        self.clearSemiDiameter = bd.max(self._height)

        # Record the maximum possible pupil size, this is a constant value and should not be change. 
        self._maxPupilSD = bd.max(self._height)

        # Copy the theoretical pupil size to the working size
        self._workingDepth = self._zDepth.copy()
        self._workingHeight = self._height.copy()

        self._CheckStopSize()
        self._ResetWorkingPupilSize()
        self._ResetSamplePool()


    def SetFirstElementSD(self, firstElementSD):
        """
        Set the semi-diameter of the first element, this is used to determine the maximum possible pupil size. 
        """
        self._firstElementSD = firstElementSD


    def SetPupilSize(self, semiDiameter):
        """
        Set the clear semi-diameter of the pupil, also regenerates the sample pool. This can be used to reflect the change of f-number.
        """

        # Check if the semi-diameter given is plausible
        if(semiDiameter > self._maxPupilSD):
            warnings.warn("The pupil size is larger than possible. Max possible size is used instead.")
            semiDiameter = self._maxPupilSD
        
        self.clearSemiDiameter = semiDiameter

        self._CheckStopSize()
        self._ResetWorkingPupilSize()
        self._ResetSamplePool()


    def SetPupilShape(self, pupilShape, areaRatio=1):
        """
        Given an image representing the pupil shape, reset the pupil sample points to fit the image.

        :param pupilShape: square RGB image array, with white meaning open, black indicating blacked.
        """
        self._alphaShape = pupilShape
        self._ResetSamplePool(int(4096.0/areaRatio))
        self._GenerateAccordingToAlphaShape()



    def GetMaxPupilSize(self):
        """
        Get the maximum possible diameter in mm.
        """
        return self._maxPupilSD * 2


    def GetSamplePoints(self, sampleCount=None):
        """
        Get some sample points that are on the pupil. 
        """

        # Return all the samples if no sample count is stated 
        if(sampleCount is None):
            return self._pupilPointSamples

        # Typically the sample count should be smaller than the size of the sample pool. If it is bigger, that might be a case of single point imaging, so just return a new set of big samples.
        # A year later: actually no let's just set the default to be really small and return a new one every time...
        # if(sampleCount > self._pupilPointSamples.shape[0]):
        if self._alphaShape is not None:
            self._ResetSamplePool(4096)
            self._GenerateAccordingToAlphaShape()
            return self._pupilPointSamples
        else:
            # Same as self._ResetSamplePool()
            pupilZdepth = bd.mean(self._workingDepth)
            return RandomEllipticalDistribution(
                major_axis=self.clearSemiDiameter,
                minor_axis=self.clearSemiDiameter,
                samplePoints=sampleCount,
                zDepth=pupilZdepth,
                groupByPoint=True)

        # selectedIndices = RNG.choice(self._pupilPointSamples.shape[0], sampleCount, replace=False)
        # return self._pupilPointSamples[selectedIndices]


    def GetEvenSamplePoints(self):
        """
        This creates a set of evenly distributed points at the pupil plane.  
        """
        pupilZdepth = bd.mean(self._workingDepth)

        return CircularDistribution(
            radius=self.clearSemiDiameter, 
            zDepth=pupilZdepth)


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


    def DrawSurface(self, overrideColor=None, smoothPoints=True):
        """
        Draw the pupil surface at the given aperture (f-number). 

        :param overrideColor: color to override the default blue color.
        :param smoothPoints: smooth the points using a Gaussian filter. 
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
            smoothDepth = self._workingDepth
            if(smoothPoints):
                smoothDepth = GaussianSmooth(self._workingDepth)
            DrawPupil(self._workingHeight, smoothDepth, surfaceColor=wlColor)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetWorkingPupilSize(self):
        """
        Reset the working pupil size to the current clear semi-diameter.
        """

        # Check if the current clear semi-diameter is plausible. Sometimes if the lens is wide open, the direct calculated value can be bigger than the maximum possible size.
        if(self.clearSemiDiameter > self._maxPupilSD):
            self.clearSemiDiameter = self._maxPupilSD

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

        :param poolSize: number of points in the sample pool. This should be set to a very high value to ensure evener sampling.
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

        
    def _CheckStopSize(self):
        """
        Ideally, the entrance pupil is defined by the aperture stop, i.e., the physical diaphragm. However, in some cases, when the 1st surface is smaller than the pupil, the entrance pupil will not be able to project fully onto the 1st surface. This method checks if the 1st surface is smaller than the pupil and adjust the max pupil size accordingly.
        """
        
        if(self._firstElementSD is None):
            return

        # When pupil is smaller than 
        if(self._maxPupilSD > self._firstElementSD):
            self._maxPupilSD = self._firstElementSD


    def _GenerateAccordingToAlphaShape(self):
        # Convert to grayscale mask (0–1 range)
        if self._alphaShape.ndim == 3:
            gray = self._alphaShape[..., :3].mean(axis=-1)
        else:
            gray = self._alphaShape
        gray = gray.astype(float)
        gray /= gray.max() if gray.max() > 0 else 1.0

        H, W = gray.shape
        center = (W - 1) / 2.0
        radius = W / 2.0  # diameter = image size

        # Project each sample to image coordinates.
        # Pupil samples are stored as (x,y,z); use x,y normalized to semi-diameter.
        samples = self._pupilPointSamples
        x = samples[:, 0]
        y = samples[:, 1]

        # Scale coordinates from physical units to pixel indices.
        # clearSemiDiameter ↔ radius pixels.
        u = (x / self.clearSemiDiameter) * radius + center
        v = (-y / self.clearSemiDiameter) * radius + center  # flip y for image coordinates

        # Round to nearest pixel and keep those within bounds.
        ui = bd.clip(bd.round(u).astype(int), 0, W - 1)
        vi = bd.clip(bd.round(v).astype(int), 0, H - 1)

        # Determine openness from the grayscale mask.
        # Points with value < 0.5 are considered blocked.
        mask_val = gray[vi.get() if hasattr(vi, "get") else vi,
        ui.get() if hasattr(ui, "get") else ui]
        open_mask = mask_val > 0.5

        # Filter out blocked points.
        self._pupilPointSamples = samples[open_mask]

        # Optionally update the effective clearSemiDiameter based on open region.
        self.clearSemiDiameter = self._maxPupilSD