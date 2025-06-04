


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
        
        """This is the distance calculated from zArray and zDepthMappingRange."""
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


        """An array of same size with the number of point sources. Each entry in this array corresponds to the """
        self.jitterPerPoint = None


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
        """
        Update the depth of the

        """
        if(newRange is None):
            newRange = self.zDepthMappingRange

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

        # Object space in world coordinate is negative, thus need to make sure the near clip is also a negative number
        if (self.nearClipping > 0 ):
            self.nearClipping = - self.nearClipping

        self.UpdateDepthRange()

        self._GeneratePolarPointSources()


    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.jitterPerPoint)

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

        # zClipDist is an array representing the
        zClipDist = self._zClipDistance(half_horizontal, half_vertical, sampleY, sampleX, self.nearClipping)

        D = bd.swapaxes(self.zDistance, 0, 1) + zClipDist #self.nearClipping

        self.jitterPerPoint = self._AngularJitter(half_horizontal, half_vertical, sampleY, sampleX, D)

        # Stack the three components into a (sampleX, sampleY, 3) array.
        coordinates = bd.stack([theta_x, theta_y, D], axis=-1)

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
        self.pointSource.isCartesian = False


    def _zClipDistance(self, horizontalHalfRad, verticalHalfRad,
                       h_steps, v_steps, clipDepth):
        """
        Calculate the distance from the camera pivot to the source points' projection on the near clipping plane.
        The rectangle sits in the XZ-plane at y = 0.
        Its corners subtend ±(horizontal_FOV/2) in X and ±(vertical_FOV/2) in Z
        from the origin *when projected onto a plane located at z = plane_z*.

        :param horizontalHalfRad: Half horizontal field-angle (rad).
        :param verticalHalfRad: Half vertical field-angle (rad).
        :param h_steps: Sub-divisions along the horizontal direction (number of intervals).
        :param v_steps: Sub-divisions along the vertical   direction (number of intervals).
        :param clipDepth: The Z-coordinate (“depth”) where the rectangle plane is located. All grid points share this z.

        :return distances: Euclidean distance from P to each rectangle node.
        """

        if(clipDepth >= 0): clipDepth = -clipDepth

        theta_h = bd.linspace(-horizontalHalfRad, horizontalHalfRad, h_steps )  # shape (h_steps,)
        theta_v = bd.linspace(-verticalHalfRad, verticalHalfRad, v_steps )  # shape (v_steps,)

        # 2. Meshgrid (X uses theta_h, Z uses theta_v)
        TH, TV = bd.meshgrid(theta_h, theta_v, indexing='xy')  # TH,TV shape (v, h)

        # 3. Convert angular deflection to X and Z on plane_z
        #    x = (plane_z) * tan(theta_h)
        #    z = plane_z + (plane_z) * tan(theta_v)
        #    (y is zero because rectangle lies in XZ plane)
        X = clipDepth * bd.tan(TH)  # shape (v, h)
        Y = clipDepth * (1 + bd.tan(TV))  # add base plane depth
        Z = bd.zeros_like(X)  # same shape

        # 4. Euclidean distance to P = (0,0,z0)
        distances = bd.sqrt(X**2 + Y**2 + clipDepth**2)  # y-term is zero

        return distances


    def _AngularJitter(self, horizontalHalfRad, verticalHalfRad,
                       h_steps, v_steps, perPointDistance):

        # angular pitch of a single pixel
        #  (use max(n-1,1) to avoid division by zero on 1-pixel edges)
        dtheta_h = (horizontalHalfRad * 2.0) / max(h_steps - 1, 1)
        # dtheta_v = (verticalHalfRad * 2.0) / max(v_steps - 1, 1)

        # ± jitter range that keeps a ray inside its pixel’s solid-angle
        jitter_h = perPointDistance * bd.tan(dtheta_h * 0.5)  # shape (v, h)
        # jitter_v = perPointDistance * bd.tan(dtheta_v * 0.5)  # shape (v, h)

        # flatten to 1-D so index order matches the flattened point list
        jitter_h_flat = bd.swapaxes(jitter_h, 0, 1).reshape(-1)
        # jitter_v_flat = bd.swapaxes(jitter_v, 0, 1).reshape(-1)

        # Horizontal and vertical angular resolution should be the same, a single return should suffice in most cases.
        return jitter_h_flat



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