

import PIL.Image
import matplotlib.pyplot as plt

# For whatever reasonimport is not working normally on my machine, had to use this ugly if else shit instead 
if __name__ == "__main__":
    from Points import PointsSource
else:
    from ObjectSpace.Points import PointsSource

from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE, PRECISION_TYPE, UP_DIR, Axis
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor
from Util.Misc import Magnitude, ArrayRotate
from Raytracing.RayBatch import RayBatch


# This class is very much an inherited class from PointSource 
# But for easier implmentation they are still separated. 

class Image2D:
    def __init__(self):
        """RGB array directly decoded from the file represneting the image"""
        self.image = None 


        """Original image file"""
        self._file = None 


        """Point source object built from the image"""
        self.pointSource = None


        """Unsigned unit in mm. If anchors are not explicitly stated, assume image at infinity"""
        self.distance = INFINITY


        """Unsigned unit in degree. If anchors are not explicitly stated, assume image in 3D fills a horizontal angle of view. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40
        # Note that since this AoV describes the image and not the lens, decreasing this attribute will make the image smaller, as if the lens is having a higher AoV. 


        """4 points data in Vec3. The 4 anchor points that pins the image in 3D space """
        self.pointAnchor = None 
        

        self.imageCenter = None 


        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


        """Height/width of each pixel, assuming square pixels"""
        self.pixelPitch = None 


    def LoadFrom8bit(self, imgPath):
        """
        For common 8 bit image formats like jpg, bmp, and png. If a png is not 8 bit, do not use this method. Find the right bit depth method instead. 
        """

        # Read and save the original 
        self._file = PIL.Image.open(imgPath).convert("RGB")

        # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._file.height * (self.imageDimensionOverride / self._file.width))
            imageFile = self._file.resize((self.imageDimensionOverride, newHeight))
        else:
            imageFile = self._file

        # Convert into array format 
        self.image = bd.array(imageFile)

        # Normalize into [0, 1 range], this is where 8 bit kicks in 
        self.image = self.image.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        self._GeneratePointSources()


    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.pixelPitch)


    def GenerateSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=5):
        """
        This generate a series of spots from axis to off axis. 
        The outer-most is defined by x and y field anfle. 
        """
        self.pointSource = PointsSource()
        self.pointSource.GenerateSpots(xAngle, yAngle, dist, sampleField)


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


    def GetSampleRatios(self):
        
        return self.pointSource.GetSampleRatios()


    def DrawImage(self):
        """
        Draw the points sources in 3D space with corresponding colors.
        """
        DrawPointsPerColor(self.pointSource.Position(), self.pointSource.Color())


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _GeneratePointSources(self):
        """
        Using the RGB and position data, generate a point source object that cooresponds to all the pixel/samples from the image input. 
        """

        # Create the 4 anchor points if they are not defined 
        if(self.pointAnchor is None):
            self._CreateAnchors()

        # This method of updating pixel pitch only works when the image is a spatial rectangle, is it stretches, then this will become uneven 
        self.pixelPitch = Magnitude(self.pointAnchor[1]-self.pointAnchor[0]) / self.imageDimensionOverride

        sampleX = self.image.shape[1]
        sampleY = self.image.shape[0]

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
        gridColors = self.image.reshape(sampleY * sampleX, 3)

        # Concatenate the position and color 
        gridPositions = bd.concatenate([gridPositions, gridColors], axis=1)

        self.pointSource = PointsSource(gridPositions)


    def _CreateAnchors(self, zDist=None):

        # Infinty is not really workable, replace it with an approximation
        if (self.distance is INFINITY):
            zDist = -FAR_DISTANCE
        else:
            zDist = -bd.array(self.distance)

        rad = bd.deg2rad(self.horizontalAoV) / 2
        halfX = bd.abs(bd.tan(rad) * zDist)
        halfY = halfX * bd.abs(self._file.height / self._file.width)

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

    img = Image2D()
    img.imageDimensionOverride = 100 
    img.distance = bd.array(100)
    img.LoadFrom8bit(r"resources/Arrow.png")
    
    RB = img.EmitSamplesToward( targets)

    for i in range(len(RB.value)):
        print(RB.value[i])

    SetUnifScale(250)
    AddXYZ()
    DrawRaybatch(RB, length=50)
    DrawPoints(targets)
    img.DrawImage()
    plt.show()

    


if __name__ == "__main__":
    main() 