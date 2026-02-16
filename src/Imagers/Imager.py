

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


from Util.ColorWavelength import WavelengthToRGB
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import ZERO, ONE
from Util.PltPlot import Reset2D, DrawRaybatch


class Imager():
    """
    Standard virtual imager with no optical effect. 
    """
    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):
        self.rayBatch = None 
        self.BFD = bfd  # Back focal distance. Sensor distance from last element's vertex 
        self.width = w
        self.height = h 
        self.horizontalPx = horiPx # Must be int 
        self.verticalPx = None  # Must be int 

        self._lensLength = 0 # Length of the lens in front of the imager 
        self._zPos = 0 

        self.rayPath = None 

        # TODO: add X and Y flip for mirror and prism 

        self._Start()

    def _Start(self):
        if (self.verticalPx == None):
            self.verticalPx = int((self.height / self.width ) * self.horizontalPx)


    def SetLensLength(self, length):
        """
        Set the length of the lens from first vertex to the last vertex, this info is used to calculate the cumulative thickness of the imager z position. 

        :param length: length of the lens. 
        """
        self._lensLength = length 
        self._zPos = length + self.BFD 


    def GetZPos(self):
        return self._zPos
    

    def IntegralRays(self, raybatch, baseImg=None, valueClamp=None):
        self.rayBatch = raybatch
        self.rayPath = [bd.copy(self.rayBatch.Position())]

        self._ImagePlaneIntersections() 
        return self._integralRays(baseImg=baseImg, valueClamp=valueClamp) 


    # ==================================================================


    """ ============================================================ """
    # ==================================================================


    def _ImagePlaneIntersections(self):
        """
        Calculate the intersections between rays (vectors from points) and a 3D plane in square shape.
        :param surfaceIndex: the index of the surface to intersect. 
        """
        # TODO: This does not seem to be working.
        
        ray_positions = self.rayBatch.Position()
        ray_directions = self.rayBatch.Direction()

        # TODO: add tilt shift support here
        imager_normal = bd.array([ZERO, ZERO, -ONE])
        plane_point = bd.array([ZERO, ZERO, self._zPos])
        
        # Calculate d (the offset from the origin in the plane equation ax + by + cz + d = 0)
        d = -bd.dot(imager_normal, plane_point)

        # Calculate dot product of direction vectors with the plane normal
        denom = bd.dot(ray_directions, imager_normal)
        
        # Avoid division by zero (for parallel vectors)
        valid_rays = (denom != 0)

        # For valid rays, calculate t where the intersection occurs
        t = -(bd.dot(ray_positions, imager_normal) + d) / denom
        
        # Calculate the intersection points
        intersection_points = ray_positions + t[:, bd.newaxis] * ray_directions

        # Find the rays that fall out of th image plane 
        outOfBoundInd = (intersection_points[:, 0] > (self.width/2)) | \
            (intersection_points[:, 0] < (-self.width/2)) | \
            (intersection_points[:, 1] > (self.height/2)) | \
            (intersection_points[:, 1] < (-self.height/2)) 
        
        # Only replace the in bound ray positions 
        ray_positions[~outOfBoundInd] = intersection_points[~outOfBoundInd]

        self.rayBatch.SetPosition(ray_positions)

        DrawRaybatch(self.rayBatch, length=10)
        plt.draw()
        plt.pause(10)

        
    def _integralRays(self, bitDepth = 8, plotResult = False, baseImg=None, valueClamp=None):
        """
        Taking integral over the rays arriving at the image plane. 

        :param primaries:
        :param secondaries:
        :param UVIRcut:
        :param bitDepth: image bitdepth.
        :param plotResult: whether to show the resulting plot or not. 
        :param baseImg: if not null, the generated image will be added onto this base image. 
        :param valueClamp: for spot simulation, normalization based on max can be inaccurate. This value is for manually override the max value for clamping. The higher it is, the darker the spot. 
        """

        try:
            self.rayBatch.SanitizePolarization()
        except Exception: pass

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
        radiantGridR = bd.zeros( (self.horizontalPx, self.verticalPx) , dtype=bd.float64)
        radiantGridG = bd.zeros( (self.horizontalPx, self.verticalPx) , dtype=bd.float64)
        radiantGridB = bd.zeros( (self.horizontalPx, self.verticalPx) , dtype=bd.float64)

        # Find all wavelengths 
        wavelengths = bd.unique(rayWavelength)
        rayHitIsolate = bd.isclose(self.rayBatch.value[:, 2], self._zPos)
        for wavelength in wavelengths:
            RGB = WavelengthToRGB(wavelength)
            wavelengthIsolate = bd.isclose(self.rayBatch.value[:, 6], wavelength)
            rayPosChannel = bd.floor(
                self.rayBatch.Position()[bd.where(rayHitIsolate & wavelengthIsolate)] / pxPitch + pxOffset).astype(int)
            radiantsChannel = self.rayBatch.Radiance()[bd.where(rayHitIsolate & wavelengthIsolate)]
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
            rgb_image = baseImg + bd.stack((red_channel, green_channel, blue_channel), axis=-1)

        if(backend_name == 'cupy'):       
            rgb_image = bd.asnumpy(rgb_image)

        if (plotResult):
            
            Reset2D()
            plt.imshow(rgb_image)
            #plt.colorbar()  # Optional: Add a colorbar to show intensity values
            plt.show()

        return rgb_image


        def _integralRaysChannelBased(self, bitDepth = 8, plotResult = False, baseImg=None, valueClamp=None):


            pass



def main():
    imager = Imager() 
    imager.Test()


if __name__ == "__main__":
    main()