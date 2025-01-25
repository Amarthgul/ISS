

import matplotlib.pyplot as plt


from Surfaces import Surface, FieldStopType
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.PltPlot import Reset2D, DrawPlane
from Util.Globals import INFINITY, ZERO, ONE, TWO
from Util.Misc import PointsInTriangle, WavelengthToRGB

class StdImager(Surface):
    """
    Standard virtual imager with no optical effect. 
    """
    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 300):

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
        self.rayBatch.SetPosition(intersections)

        return self.rayBatch, _tir, _vig


    def IntegralRays(self, raybatch, baseImg=None,):

        self.rayBatch = raybatch

        self._integralRays(baseImg=baseImg)

        


    def DrawSurface(self):
        DrawPlane(self.gatePoints, color='black')


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _FieldStopMask(self, intersections):
        """
        Given the intersections, filter out the ones that are outside of the field stop. 
        """

        sideA = PointsInTriangle(intersections, self.gatePoints[0], self.gatePoints[1], self.gatePoints[2])
        sideB = PointsInTriangle(intersections, self.gatePoints[0], self.gatePoints[2], self.gatePoints[3])
        return sideA | sideB

    
    def _integralRays(self, bitDepth=8, plotResult=True, baseImg=None, valueClamp=None):
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
            radiantsChannel = self.rayBatch.Radiant()[bd.where(rayHitIsolate & wavelengthIsolate)]
            rChannel = radiantsChannel * RGB[0]
            gChannel = radiantsChannel * RGB[1]
            bChannel = radiantsChannel * RGB[2]
            bd.add.at(radiantGridR, (rayPosChannel[:, 0], rayPosChannel[:, 1]), rChannel)
            bd.add.at(radiantGridG, (rayPosChannel[:, 0], rayPosChannel[:, 1]), gChannel)
            bd.add.at(radiantGridB, (rayPosChannel[:, 0], rayPosChannel[:, 1]), bChannel)
        
        maxValue = bd.max(bd.array([bd.max(radiantGridR), bd.max(radiantGridG), bd.max(radiantGridB)]))
        bits = 2.0**bitDepth-1

        if(valueClamp is None):
            # Suitable for image sim 
            scaleRatio = (bits / maxValue) 
        else:
            # Spot sim 
            scaleRatio = (bits / valueClamp)

        red_channel = bd.clip(radiantGridR*scaleRatio, 0, bits) 
        green_channel = bd.clip(radiantGridG*scaleRatio, 0, bits)  
        blue_channel = bd.clip(radiantGridB*scaleRatio, 0, bits)

        
        # Ensure each channel is in the range [0, 255] and convert to uint8
        # TODO: edit this to reflect bitdepth 
        red_channel = red_channel.astype(bd.uint8)
        
        green_channel = green_channel.astype(bd.uint8)
        blue_channel = blue_channel.astype(bd.uint8)

        if(baseImg == None):
            # Stack the channels along the third axis to form an RGB image
            rgb_image = bd.stack((red_channel, green_channel, blue_channel), axis=-1)
        else:
            # In case this is an iterative call with an already formed image 
            rgb_image += bd.stack((red_channel, green_channel, blue_channel), axis=-1)

        if (plotResult):
            if(backend_name == 'cupy'):       
                rgb_image = bd.asnumpy(rgb_image)
            Reset2D()
            plt.imshow(rgb_image)
            #plt.colorbar()  # Optional: Add a colorbar to show intensity values
            plt.show()

        return rgb_image













