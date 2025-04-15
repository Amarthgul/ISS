


import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR

import sys
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from .Points import PointsSource
from .Images import Image2D


from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE, KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor, RemoveBG
from Util.Misc import Magnitude, ArrayRotate, PolarToCartesian, RectPath
from Raytracing.RayBatch import RayBatch








class Image2DVariDepth(Image2D):

    def __init__(self):
        super().__init__()

        """RGB array of the image"""
        self.rgbArray = None 

        """Z depth array of the image. This is the direct value of the image representing the z depth, not the acutal physical distance."""
        self.zArray = None 
        
        """This is the distnace calculated from zArray and zDepthMappingRange."""
        self.zDistance = None 

        """Because this is a secondary imaging process, an angle of view of the image source is needed. Value is unsigned unit in degree. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40


        """Master image file. For EXR this could include the alpha and the z depth"""
        self._fileMaster = None 

        """Separate file for the Z depth"""
        self._fileZ = None 

        """Point source object built from the image"""
        self.pointSource = None


        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


        """Z depth read form the input are in the range of [0, 1]. However, for actual imaging, this apparently is a not a valid distance range. This attribute is used to map the z depth into a more realistic range. By default, the range is set to 1.5m to 500m, i.e., typicaly portrait distance to infinity"""
        self.zDepthMappingRange = bd.array([KNOB_DISTANCE, FAR_DISTANCE])


        """For z-depth from rendered images, it typically means the distance from the object to the near clipping plane. To reconstruct the scene, it is thus needed to calculate the cone formed by the near clipping plane as well. 
        This property marks the distance from the near clipping plane to the camera, i.e.e, the (0, 0, 0) point. This should be an unsigned value."""
        self.nearClipping = 0



    def LoadFrom8bit(self, rgbImgPath, zImgPath=None):
        """
        For common 8 bit image formats like jpg, bmp, and png. Since these images tppically only contain the RGB information, the z depth information have to be read from a separate file. 

        :param rgbImgPath: Path to the RGB image file
        :param zImgPath: Path to the Z depth image file
        """

        self.LoadFrom8bitRGB(rgbImgPath)

        if(zImgPath is not None):
            self.LoadFrom8bitZ(zImgPath)

        self.Refresh()


    def LoadFrom8bitRGB(self, rgbImgPath):
        """
        Load an RGB image from the given path. 
        """
        rgbImgPath = RectPath(rgbImgPath)

        # Read and save the master file  
        self._fileMaster = PIL.Image.open(rgbImgPath).convert("RGB")
         # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            RGBImageFile = self._fileMaster.resize((self.imageDimensionOverride, newHeight))
        else:
            RGBImageFile = self._fileMaster

        # Convert into array format 
        self.rgbArray = bd.array(RGBImageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in 
        self.rgbArray = self.rgbArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)


    def LoadFrom8bitZ(self, zImgPath):

        zImgPath = RectPath(zImgPath)

        # Read and save the z depth file 
        self._fileZ = PIL.Image.open(zImgPath).convert("L")

        # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            ZImageFile = self._fileZ.resize((self.imageDimensionOverride, newHeight))
        else:
            ZImageFile = self._fileZ

        # Convert into array format 
        self.zArray = bd.array(ZImageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in 
        self.zArray = self.zArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        self.UpdateDepthRange()


    def UpdateDepthRange(self, newRange=None):

        if(newRange is not None):
            assert len(newRange) == 2, "The new range should be a list of two values"
            if(newRange[0] > newRange[1]):
                self.zDepthMappingRange = bd.array([newRange[1], newRange[0]])
            else:
                self.zDepthMappingRange = bd.array(newRange)

        deltaRange = self.zDepthMappingRange[1] - self.zDepthMappingRange[0]

        self.zDistance = self.zArray * deltaRange + self.zDepthMappingRange[0]

        # Here the z Distance is unsigned, to make it work in the system, they need to be inverted. 
        self.zDistance = -self.zDistance


    def Refresh(self):
        """
        Manually refresh the parameters. Remap the depth and recreate the point sources. 
        """

        self.UpdateDepthRange()

        self._GeneratePolarPointSources()


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _GeneratePolarPointSources(self):
        """
        Generate point sources from the image where each pixel is represented 
        in polar coordinates: [theta_x, theta_y, D, R, G, B]. 
        Here, theta_x and theta_y are field angles (in radians) relative to the optical axis,
        and D is the distance from the front vertex (a constant value, self.distance).
        """
        # Get image dimensions from the normalized RGB array.
        sampleY, sampleX, _ = self.rgbArray.shape  # note: PIL images are (height, width, channels)
        
        # Create interpolation factors for horizontal (u) and vertical (v) directions.
        u = bd.linspace(0, 1, sampleX)
        v = bd.linspace(0, 1, sampleY)
        
        # Create a meshgrid; use "ij" indexing so that U varies along width and V along height.
        U, V = bd.meshgrid(u, v, indexing="ij")  # U and V shape: (sampleX, sampleY)
        
        # Convert the horizontal angle of view (in degrees) to radians.
        horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
        half_horizontal = horizontalAoV_rad / 2.0
        
        # Compute vertical AoV from the image aspect ratio.
        verticalAoV_rad = horizontalAoV_rad * (sampleY / sampleX)
        half_vertical = verticalAoV_rad / 2.0
        
        # For each pixel column, compute theta_x in the range [-half_horizontal, half_horizontal]
        theta_x = -half_horizontal + U * horizontalAoV_rad
        # For each pixel row, compute theta_y in the range [-half_vertical, half_vertical]
        theta_y = -half_vertical + V * verticalAoV_rad
        
        D = bd.swapaxes(self.zDistance, 0, 1)  
        # Stack the three components into a (sampleX, sampleY, 3) array.
        coordinates = bd.stack([theta_x, theta_y, D], axis=-1)

        # Convert the angle field to Cartesian coordinates 
        coordinates = PolarToCartesian(coordinates, angleInRad=True)

        # The grid was built with U, V of shape (sampleX, sampleY), but typically 
        # we want the final list of points to be organized per pixel (row-major order).
        # Swap axes if needed, then reshape to (sampleX*sampleY, 3)
        coordinates = bd.swapaxes(coordinates, 0, 1)  # now shape: (sampleY, sampleX, 3)
        coordinates = coordinates.reshape(sampleY * sampleX, 3)
        
        # Get the RGB color for each pixel and reshape it accordingly.
        colors = self.rgbArray.reshape(sampleY * sampleX, 3)
        
        # Concatenate the polar coordinates and color channels:
        points = bd.concatenate([coordinates, colors], axis=1)
        
        # Finally, create a new PointsSource with these points.
        self.pointSource = PointsSource(points)




def main():
    
    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    img = Image2DVariDepth()
    img.imageDimensionOverride = 200 
    img.zDepthMappingRange = bd.array([500, 1000])

    img.LoadFrom8bit(r"resources/DualTest_RGB.png", r"resources/DualTest_Z.png")

    img.DrawImage()

    RemoveBG()
    SetUnifScale(1000)
    plt.show()




if __name__ == "__main__":
    main()