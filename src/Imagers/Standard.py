

import matplotlib.pyplot as plt

from Surfaces.Surface import FieldStopType, Surface
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.PltPlot import Reset2D, DrawPlane
from Util.Globals import INFINITY, ZERO, ONE, TWO, Axis
from Util.ColorWavelength import WavelengthToRGB
from Util.Misc import PointsInTriangle, NumpyConversion

class StdImager(Surface):
    """
    Standard virtual imager with no optical effect. 
    """
    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):

        # Raduis is infinity, no thickness, clear semi-diameter is infinity
        super().__init__(INFINITY, ZERO, INFINITY)

        self.fType = FieldStopType.Rectangular

        self.rayBatch = None 
        self.BFD = bfd  # Back focal distance. Sensor distance from last element's vertex 
        
        self.width = bd.array(w)          # Physical size in mm
        self.height = bd.array(h)         # Physical size in mm    

        self.horizontalPx = horiPx  # Unitless int 
        self.verticalPx = None      # Unitless int 

        """Points reprensenting the 4 corners of the rectangular field stop"""
        self.gatePoints = None

        self.rayPath = None 


        # If gate points are not set, then assume it is flat and perpendicular to the optical axis, then use the 2 following properties to calculate the gate points
        self._lensLength = 0 # Length of the lens in front of the imager
        self._zPos = 0
        

        self._Start()


    def _Start(self):
        if (self.verticalPx is None):
            # If vertical pixel is not set, calculate it from aspect ratio
            self.verticalPx = int((self.height / self.width ) * self.horizontalPx)

        
    def SetLensLength(self, length):
        """
        Set the length of the lens from first vertex to the last vertex, this info is used to calculate the cumulative thickness of the imager z position. 

        :param length: length of the lens. 
        """
        self._lensLength = length 
        self._zPos = length + self.BFD 

        if(self.gatePoints is None):
            self.gatePoints = bd.array([
                [ self.width/2,  self.height/2, self._zPos], 
                [-self.width/2,  self.height/2, self._zPos], 
                [-self.width/2, -self.height/2, self._zPos], 
                [ self.width/2, -self.height/2, self._zPos]])
            
        self.frontVertex = bd.array([ZERO, ZERO, self._zPos])


    def IntersectRays(self, raybatch):
        self.rayBatch = raybatch
        intersections, _tir, _vig = self.Intersection(self.rayBatch)
        self.rayBatch.Mask(~_vig)
        self.rayBatch.SetPosition(intersections)
        

        return self.rayBatch, _tir, _vig


    def IntegralRays(self, raybatch, baseImg=None,):

        self.rayBatch = raybatch

        return self._integralRays(baseImg=baseImg)


    def DrawSurface(self):
        DrawPlane(self.gatePoints, color='black')


    def AccquireEmpty(self):
        """
        Accquire an empty image array. When converted, this will be a black and blank image. 
        """
        # TODO: add more bitdepth support 
        imgAry = bd.zeros((self.horizontalPx , self.verticalPx, 3),  dtype=bd.uint8)
        
        # if(backend_name == 'cupy'):       
        #     imgAry = bd.asnumpy(imgAry)

        return imgAry


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _FieldStopMask(self, intersections):
        """
        Given the intersections, filter out the ones that are outside of the field stop. 
        """
        # If tilt shift is involved, consider using the triangle intersection check 
        heightMask = (intersections[:, Axis.Y.value] > self.height/2 ) |(intersections[:, Axis.Y.value] < -self.height/2)
        widthMask = (intersections[:, Axis.X.value] > self.width/2 ) | (intersections[:, Axis.X.value] < -self.width/2)

        return ~(heightMask | widthMask)

    
    def _integralRays(self, bitDepth=8, baseImg=None, valueClamp=None):
        """
        Taking integral over the rays arriving at the image plane. 

        :param bitDepth: image bitdepth.
        :param plotResult: whether to show the resulting plot or not. 
        :param baseImg: if not null, the generated image will be added onto this base image. 
        :param valueClamp: for spot simulation, normalization based on max can be inaccurate. This value is for manually override the max value for clamping. The higher it is, the darker the spot. 
        """

        pxPitch = self.width / self.horizontalPx 
        pxOffset = bd.array([self.horizontalPx/2, self.verticalPx/2, 0])

        # Find the rays that arrived at the the image plane 
        rayHitIndex = bd.where(bd.isclose(self.rayBatch.value[:, 2], self._zPos))

        # Translate the intersections from 3D image space to 2D pixel-based space
        rayPos = self.rayBatch.Position()[rayHitIndex] / pxPitch + pxOffset
        rayWavelength = self.rayBatch.Wavelength()[rayHitIndex] 

        # Convert ray position into pixel position 
        rayPos = bd.floor(rayPos).astype(int)
        # Create pixel grid 
        radiantGridR = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridG = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridB = bd.zeros( (self.horizontalPx, self.verticalPx) )

        # Find all wavelengths 
        wavelengths = bd.unique(rayWavelength)
        rayHitIsolate = bd.isclose(self.rayBatch.value[:, 2], self._zPos)
        for wavelength in wavelengths:
            RGB = WavelengthToRGB(wavelength)
            wavelengthIsolate = bd.isclose(self.rayBatch.value[:, 6], wavelength)
            rayPosChannel = bd.floor(
                self.rayBatch.Position()[bd.where(rayHitIsolate & wavelengthIsolate)] / pxPitch + pxOffset).astype(int)[:, :2]
            radiantsChannel = self.rayBatch.PolarizedRadiance()[bd.where(rayHitIsolate & wavelengthIsolate)]
            rChannel = radiantsChannel * RGB[0]
            gChannel = radiantsChannel * RGB[1]
            bChannel = radiantsChannel * RGB[2]
            bd.add.at(radiantGridR, (rayPosChannel[:, 0], rayPosChannel[:, 1]), rChannel)
            bd.add.at(radiantGridG, (rayPosChannel[:, 0], rayPosChannel[:, 1]), gChannel)
            bd.add.at(radiantGridB, (rayPosChannel[:, 0], rayPosChannel[:, 1]), bChannel)
        
        # Stack the channels together as a "latent image"
        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)
        
        # Monte Carlo addition 
        if(baseImg is not None):
            rgb_image = baseImg + rgb_image

        #maxValue = bd.max(rgb_image)
        #print("max value: ", maxValue)

        return rgb_image













