

import PIL.Image
import matplotlib.pyplot as plt
import Imath
import OpenEXR

import sys
import os


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from .Points import PointsSource

from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE, PRECISION_TYPE, UP_DIR, Axis
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor
from Util.Misc import Magnitude, ArrayRotate, RectPath
from Raytracing.RayBatch import RayBatch





class Image2D:
    def __init__(self):
        """RGB array directly decoded from the file representing the image"""

        # This class is very much an inherited class from PointSource 
        # But for easier implementation they are still separated. 
        
        self.rgbArray = None 

        """Original image file"""
        self._fileMaster = None 

        """Point source object built from the image"""
        self.pointSource = None

        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.pixelPitch)


    def GenerateSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=5):
        """
        This generate a series of spots from axis to off axis. 
        The outer-most is defined by x and y field anfle. 
        """
        self.pointSource = PointsSource()
        self.pointSource.GenerateSpots(xAngle, yAngle, dist, sampleField)


    def GetSampleRatios(self):
        
        return self.pointSource.GetSampleRatios()


    def DrawImage(self):
        """
        Draw the points sources in 3D space with corresponding colors.
        """
        DrawPointsPerColor(self.pointSource.Position(), self.pointSource.DisplayColor())



class Image2DFlat(Image2D):
    def __init__(self):
        super().__init__()


        """Unsigned unit in mm. If anchors are not explicitly stated, assume image at infinity"""
        self.distance = INFINITY


        """Unsigned unit in degree. If anchors are not explicitly stated, assume image in 3D fills a horizontal angle of view. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40
        # Note that since this AoV describes the image and not the lens, decreasing this attribute will make the image smaller, as if the lens is having a higher AoV. 


        """4 points data in Vec3. The 4 anchor points that pins the image in 3D space """
        self.pointAnchor = None 
        

        self.imageCenter = None 
        

        """Height/width of each pixel, assuming square pixels"""
        self.pixelPitch = None


        self._opacity = None
        self._opacityArray = None


    def LoadFrom8bit(self, imgPath):
        """
        For common 8 bit image formats like jpg, bmp, and png. If a png is not 8 bit, do not use this method. Find the right bit depth method instead. 
        """

        # Read and save the original
        imgPath = RectPath(imgPath)
        self._fileMaster = PIL.Image.open(imgPath).convert("RGB")

        self._Update()


    def LoadFrom8BitPNG(self, imgPath):
        # Read and save the original
        imgPath = RectPath(imgPath)
        self._fileMaster = PIL.Image.open(imgPath).convert("RGBA")
        # if self._fileMaster.mode != "RGBA":
        #     self._fileMaster = self._fileMaster.convert("RGBA")

        r, g, b, self._opacity = self._fileMaster.split()

        self._fileMaster = self._fileMaster.convert("RGB")


        self._Update()


    def SetupTransitionTest(self, rotateDegree=45, scale=2):
        """
        This method adjusts the anchor points thus tilting the image. The tilted image can then be used to test transition. 

        """
        
        # Create the 4 anchor points if they are not defined 
        if(self.pointAnchor is None):
            self._CreateAnchors(self.distance)

        self.pointAnchor = ArrayRotate(bd.pi/4, 
                                       UP_DIR, 
                                       self.imageCenter, 
                                       self.pointAnchor)


        self._GeneratePointSources()


    def LoadFromEXR(self, imgPath):
        """
        Load only the RGB info from an EXR image. Other channels are ignored.
        """
        exr = OpenEXR.InputFile(imgPath)

        # EXR header tells us the image size
        header = exr.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # EXR stores channels as strings like "R", "G", "B"
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        # Read raw channel strings
        r_str = exr.channel('R', FLOAT)
        g_str = exr.channel('G', FLOAT)
        b_str = exr.channel('B', FLOAT)

        # Convert to float32 NumPy arrays
        r = bd.frombuffer(r_str, dtype=bd.float32).reshape((height, width))
        g = bd.frombuffer(g_str, dtype=bd.float32).reshape((height, width))
        b = bd.frombuffer(b_str, dtype=bd.float32).reshape((height, width))

        # Stack to H×W×3
        rgb = bd.stack([r, g, b], axis=-1)
        self.rgbArray = bd.stack([r, g, b], axis=-1)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _Update(self):

        # Resize the input if needed
        if self.imageDimensionOverride is not None:
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            imageFile = self._fileMaster.resize((self.imageDimensionOverride, newHeight))
            if self._opacity is not None:
                opacityArray = self._opacity.resize((self.imageDimensionOverride, newHeight))
                self._opacityArray = bd.array(opacityArray)
        else:
            self.imageDimensionOverride = self._fileMaster.width
            imageFile = self._fileMaster

        # Convert into array format
        self.rgbArray = bd.array(imageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in
        self.rgbArray = self.rgbArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)


        self._GeneratePointSources()


    def _GeneratePointSources(self):
        """
        Using the RGB and position data, generate a point source object that cooresponds to all the pixel/samples from the image input. 
        """

        # Create the 4 anchor points if they are not defined 
        if(self.pointAnchor is None):
            self._CreateAnchors()

        # This method of updating pixel pitch only works when the image is a spatial rectangle, is it stretches, then this will become uneven 
        self.pixelPitch = Magnitude(self.pointAnchor[1]-self.pointAnchor[0]) / self.imageDimensionOverride

        sampleX = self.rgbArray.shape[1]
        sampleY = self.rgbArray.shape[0]

        u = bd.linspace(0, 1, sampleX)  # Interpolation values in x-direction
        v = bd.linspace(0, 1, sampleY)  # Interpolation values in y-direction

        # Create a meshgrid of interpolation factors
        U, V = bd.meshgrid(u, v, indexing="ij")  # Shape (sampleX, sampleY)

        # Compute the bilinear interpolation
        gridPositions = (
            (1 - U)[..., None] * (1 - V)[..., None]  * self.pointAnchor[0].reshape(1, 1, 3) +
            U[..., None] * (1 - V)[..., None]        * self.pointAnchor[1].reshape(1, 1, 3) +
            (1 - U)[..., None] * V[..., None]        * self.pointAnchor[2].reshape(1, 1, 3) +
            U[..., None] * V[..., None]              * self.pointAnchor[3].reshape(1, 1, 3)
        )  

        # The grid generated this way is transposed, thus need the axis swapped
        gridPositions = bd.swapaxes(gridPositions, 0, 1)

        # Reshape the point position and color array
        gridPositions = gridPositions.reshape(sampleY * sampleX, 3)
        gridColors = self.rgbArray.reshape(sampleY * sampleX, 3)

        if self._opacityArray is not None:
            flat_opacity = self._opacityArray.reshape(sampleY * sampleX)
            mask = flat_opacity > 0
            gridPositions = gridPositions[mask]
            gridColors = gridColors[mask]

        gridData = bd.concatenate([gridPositions, gridColors], axis=1)
        self.pointSource = PointsSource(gridData)

        # Concatenate the position and color 
        # gridPositions = bd.concatenate([gridPositions, gridColors], axis=1)
        #
        # self.pointSource = PointsSource(gridPositions)


    def _CreateAnchors(self, zDist=None):

        # Infinty is not really workable, replace it with an approximation
        if (self.distance is INFINITY):
            zDist = -FAR_DISTANCE
        else:
            zDist = -bd.array(self.distance)

        rad = bd.deg2rad(self.horizontalAoV) / 2

        halfX = bd.abs(bd.tan(rad) * zDist)
        halfY = halfX * bd.abs(self._fileMaster.height / self._fileMaster.width)

        self.pointAnchor = bd.array([
            [-halfX, -halfY, zDist], 
            [ halfX, -halfY, zDist], 
            [-halfX,  halfY, zDist], 
            [ halfX,  halfY, zDist], 
        ])

        self.imageCenter = bd.mean(self.pointAnchor, axis=0)


def main():
    
    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    img = Image2DFlat()
    img.imageDimensionOverride = 100 
    img.distance = bd.array(100)
    img.LoadFrom8bit(r"resources/Arrow.png")
    
    RB = img.EmitSamplesToward( targets)

    for i in range(len(RB.value)):
        print(RB.value[i])

    SetUnifScale(250)
    AddXYZ()
    DrawRaybatch(RB, lLength=50)
    DrawPoints(targets)
    img.DrawImage()
    plt.show()

    

if __name__ == "__main__":
    main() 

