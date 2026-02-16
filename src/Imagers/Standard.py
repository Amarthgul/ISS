

import matplotlib.pyplot as plt

from Surfaces.Surface import FieldStopType, Surface
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.PltPlot import Reset2D, DrawPlane
from Util.Globals import INFINITY, ZERO, ONE, TWO, Axis, OUTPUT_TYPE
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

        #self.rayBatch = None 
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

        # Whether to use probability density function for wavelength generation, or direct Fraunhofer line replacement
        self._usePDF = True

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

        self.Update()


    def Update(self):
        self._zPos = self._lensLength + self.BFD

        if (self.gatePoints is None):
            self.gatePoints = bd.array([
                [self.width / 2, self.height / 2, self._zPos],
                [-self.width / 2, self.height / 2, self._zPos],
                [-self.width / 2, -self.height / 2, self._zPos],
                [self.width / 2, -self.height / 2, self._zPos]])

        self.frontVertex = bd.array([ZERO, ZERO, self._zPos])


    def IntersectRays(self, raybatch):
        #self.rayBatch = raybatch
        intersections, _tir, _vig = self.Intersection(raybatch)
        raybatch.Mask(~_vig)
        raybatch.SetPosition(intersections)
        

        return raybatch, _tir, _vig


    def IntegralRays(self, raybatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):

        #self.rayBatch = raybatch

        return self._integralRays(raybatch, baseImg=baseImg, overExpNoiseRemoval=overExpNoiseRemoval, polarized=polarized)


    def DrawSurface(self):
        DrawPlane(self.gatePoints, color='black')


    def AcquireEmpty(self, dataType=OUTPUT_TYPE):
        """
        Accquire an empty image array. When converted, this will be a black and blank image. 
        """
        
        imgAry = bd.zeros((self.horizontalPx , self.verticalPx, 3),  dtype=dataType)
    
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


    def _integralRays(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):

        if self._usePDF:
            return self._integralRaysChannelBased(intersectRayBatch, baseImg, overExpNoiseRemoval,polarized)

        else:
            return self._integralRaysFraunhoferLine(intersectRayBatch, baseImg, overExpNoiseRemoval, polarized)



    def _integralRaysFraunhoferLine(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """
        Taking integral over the rays arriving at the image plane. 

        :param bitDepth: image bitdepth.
        :param plotResult: whether to show the resulting plot or not. 
        :param baseImg: if not null, the generated image will be added onto this base image. 
        :param overExpNoiseRemoval: sometimes there are single pixel of value that is way overexposed due to random errors. When overExpNoiseRemoval is set to a number, it will prune the values depending on the std of the values. 
        
        :return: a float array representing an RGB image.  
        """


        #print("Total intersect rays ", intersectRayBatch.value.shape)

        pxPitch = self.width / self.horizontalPx 
        pxOffset = bd.array([self.horizontalPx/2, self.verticalPx/2, 0])

        # Find the rays that arrived at the the image plane 
        rayHitMask = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Translate the intersections from 3D image space to 2D pixel-based space
        rayPos = intersectRayBatch.Position()[rayHitMask] / pxPitch + pxOffset
        rayWavelength = intersectRayBatch.Wavelength()[rayHitMask]

        # Convert ray position into pixel position 
        rayPos = bd.floor(rayPos).astype(int)
        # Create pixel grid 
        radiantGridR = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridG = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridB = bd.zeros( (self.horizontalPx, self.verticalPx) )

        # Isolate the rays that arrived at the imager plane 
        rayHitIsolate = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Find all wavelengths 
        wavelengths = bd.unique(rayWavelength)
        for wavelength in wavelengths:
            RGB = WavelengthToRGB(wavelength)

            # Isolate the wavelength currently dealing with 
            wavelengthIsolate = bd.isclose(intersectRayBatch.Wavelength(), wavelength)

            # Convert the position of the ray hits into an int grid by flooring them 
            rayPosChannel = bd.floor(
                intersectRayBatch.Position()[rayHitIsolate & wavelengthIsolate] / pxPitch + pxOffset).astype(int)[:, :2]
            

            radiantChannel = intersectRayBatch.PolarizedRadiance(polarized)[rayHitIsolate & wavelengthIsolate]

            # Masking out the rays outside of imager area, avoiding the numpy negative index from shifting the rays...
            in_bounds = (
                    (rayPosChannel[:, 0] >= 0) & (rayPosChannel[:, 0] < self.horizontalPx) &
                    (rayPosChannel[:, 1] >= 0) & (rayPosChannel[:, 1] < self.verticalPx)
            )
            rayPosChannel = rayPosChannel[in_bounds]
            radiantChannel = radiantChannel[in_bounds]

            #print("radiantChannel max: ", bd.max(radiantChannel), "\t\t mean", bd.mean(radiantChannel), "\t\t std ", bd.std(radiantChannel))

            # Try to remove the outlier over-exposed pixels (maybe caused by float error?)
            if((overExpNoiseRemoval is not None) and 
               (bd.max(radiantChannel) > bd.mean(radiantChannel))):
                # TODO: add a second condition for automatic check if both mean and max are the same
                radiantChannel = self._PruneHighOutliers(radiantChannel, overExpNoiseRemoval)

            rChannel = radiantChannel * RGB[0]
            gChannel = radiantChannel * RGB[1]
            bChannel = radiantChannel * RGB[2]

            bd.add.at(radiantGridR, (rayPosChannel[:, 0], rayPosChannel[:, 1]), rChannel)
            bd.add.at(radiantGridG, (rayPosChannel[:, 0], rayPosChannel[:, 1]), gChannel)
            bd.add.at(radiantGridB, (rayPosChannel[:, 0], rayPosChannel[:, 1]), bChannel)
        
        # Stack the channels together as a "latent image"
        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)
        
        # Monte Carlo addition 
        if(baseImg is not None):
            rgb_image = baseImg + rgb_image

        return rgb_image


    def _integralRaysChannelBased(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """
        Channel-based integral:
          - Uses rayBatch.Channel() (0=R, 1=G, 2=B) to directly deposit energy
          - Uses PolarizedRadiance(polarized=...) for intensity (same as _integralRays)
          - Optionally prunes over-exposed outliers (same idea as _integralRays)
          - Optionally adds onto baseImg (Monte Carlo accumulation)
        """

        pxPitch = self.width / self.horizontalPx
        pxOffset = bd.array([self.horizontalPx / 2, self.verticalPx / 2, 0])

        # Rays that arrived at the image plane
        rayHitMask = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Convert hit positions to pixel coords
        rayPos = bd.floor(intersectRayBatch.Position()[rayHitMask] / pxPitch + pxOffset).astype(int)[:, :2]

        # Radiance + channel id per ray
        radiant = intersectRayBatch.PolarizedRadiance(polarized)[rayHitMask]
        chan = intersectRayBatch.Channel()[rayHitMask].astype(int)

        # Mask out hits outside the imager area (avoid negative indexing issues)
        in_bounds = (
                (rayPos[:, 0] >= 0) & (rayPos[:, 0] < self.horizontalPx) &
                (rayPos[:, 1] >= 0) & (rayPos[:, 1] < self.verticalPx)
        )
        rayPos = rayPos[in_bounds]
        radiant = radiant[in_bounds]
        chan = chan[in_bounds]

        # Create pixel grids
        radiantGridR = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridG = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridB = bd.zeros((self.horizontalPx, self.verticalPx))

        # Deposit per channel (and prune outliers per-channel, like the wavelength loop did)
        for c, grid in ((0, radiantGridR), (1, radiantGridG), (2, radiantGridB)):
            m = (chan == c)
            if not bd.any(m):
                continue

            pos_c = rayPos[m]
            rad_c = radiant[m]

            # Try to remove the outlier over-exposed rays (same condition style as _integralRays)
            if (overExpNoiseRemoval is not None) and (bd.max(rad_c) > bd.mean(rad_c)):
                rad_c = self._PruneHighOutliers(rad_c, overExpNoiseRemoval)

            bd.add.at(grid, (pos_c[:, 0], pos_c[:, 1]), rad_c)

        # Stack to RGB image
        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)

        # Monte Carlo accumulation
        if baseImg is not None:
            rgb_image = baseImg + rgb_image

        return rgb_image


    def _PruneHighOutliers(self, arr, k=4.0):
        """
        Replace values from an array that are greater than mean + k * std.
        
        :param arr: Input array.
        :param k: Number of standard deviations above the mean to use as threshold.
            
        :return: Array whose outliers are replaced with 0.
        """
        arr = bd.asarray(arr)
        mean = bd.mean(arr)
        std = bd.std(arr)
        threshold = mean + k * std
        arr[arr >= threshold] = 0
        
        return arr










