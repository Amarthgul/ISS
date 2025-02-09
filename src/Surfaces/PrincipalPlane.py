


from Util.ColorWavelength import ColorTuplePLT, WavelengthToRGB
from Util.Misc import Normalized, ArrayMagnitude
from Util.Backend import backend as bd 
from Util.PltPlot import DrawDisk, DrawPupil, DrawPoints


from .VirtualSurface import VirtualSurface, SymmetryType


class PrincipalPlane(VirtualSurface):
    def __init__(self):
        
        self.symmetryType = SymmetryType.Axial
        # By default the pupil is axial symmetric 

        self.clearSemiDiameter = None 

        """The wavelength for the sample points"""
        self.sampleWavelength = None 

        self._height = []
        self._zDepth = []



    def AddSamplePoint(self, point):
        """
        Add a single point into the principal plane samples.
        """

        self._zDepth = bd.append(self._zDepth, point[2])
        self._height = bd.append(self._height, Normalized(bd.array([point[0], point[1]])))

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


    def GetOuterZ(self):
        """
        Find the z depth of the principle plane using the coordinate of the outer-most sample point. 
        """

        outerIndex = bd.argmax(self._height)
        return self._zDepth[outerIndex]


    def GetInnerZ(self):
        """
        Find the z depth of the principle plane using the coordinate of the inner-most sample point. 
        """

        outerIndex = bd.argmin(self._height)
        return self._zDepth[outerIndex]



    def DrawSurface(self, overrideColor=None):

        wlColor = 'b'

        if (not self.sampleWavelength == None):
            wlColor = ColorTuplePLT(WavelengthToRGB(self.sampleWavelength))
        if(not overrideColor == None):
            wlColor = overrideColor

        if(len(self._zDepth) == 1):
            # When there is only one data point,
            # Assume it is the center point on axis and use it as the overall depth 
            DrawDisk(self.clearSemiDiameter, self._zDepth[0], surfaceColor=wlColor)

        else:
            # When there are many different points for the pupil plane 
            DrawPupil(self._height, self._zDepth, surfaceColor=wlColor)


    def DrawSamplePoints(self, overrideColor=None, duplicateAxial=True):

        wlColor = 'b'

        if (not self.sampleWavelength == None):
            wlColor = ColorTuplePLT(WavelengthToRGB(self.sampleWavelength))
        if(not overrideColor == None):
            wlColor = overrideColor

        points = bd.stack(
            (bd.zeros(len(self._height)), self._height, self._zDepth), axis=-1)

        if(duplicateAxial):
            opposite = bd.stack(
                (bd.zeros(len(self._height)), -self._height, self._zDepth), axis=-1)
            points = bd.vstack((points, opposite))

        DrawPoints(points, color=wlColor)

