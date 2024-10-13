
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from Lens import * 

class Imager():
    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 300):
        self.rayBatch = None 
        self.BFD = bfd  # Back focal distance. Sensor distance from last element's vertex 
        self.width = w
        self.height = h 
        self.horizontalPx = horiPx # Must be int 
        self.verticalPx = None  # Must be int 

        self._lensLength = 0 # Length of the lens in front of the imager 
        self._zPos = 0 

        self.rayPath = None 

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
    

    def IntegralRays(self, raybatch):
        self.rayBatch = raybatch
        self.rayPath = [np.copy(self.rayBatch.Position())]

        self._ImagePlaneIntersections() 
        self._integralRays() 

    def Test(self):
        val = self._RGBToWavelength([255, 128, 10])
        print("\n\nValue: ", val)
        

    # ==================================================================
    """ ============================================================ """
    # ==================================================================


    def _ImagePlaneIntersections(self):
        """
        Calculate the intersections between rays (vectors from points) and a 3D plane in square shape.
        :param surfaceIndex: the index of the surface to intersect. 
        """
        og_positions = self.rayBatch.Position()
        og_direction = self.rayBatch.Direction()

        ray_positions = og_positions[np.where(self.rayBatch.Sequential() == 1)]
        ray_directions = og_direction[np.where(self.rayBatch.Sequential() == 1)]

        # TODO: add tilt shift support here
        imager_normal = np.array([0, 0, -1])
        plane_point = np.array([0, 0, self._zPos])
        
        # Calculate d (the offset from the origin in the plane equation ax + by + cz + d = 0)
        d = -np.dot(imager_normal, plane_point)

        # Calculate dot product of direction vectors with the plane normal
        denom = np.dot(ray_directions, imager_normal)
        
        # Avoid division by zero (for parallel vectors)
        valid_rays = (denom != 0)

        # For valid rays, calculate t where the intersection occurs
        t = -(np.dot(ray_positions, imager_normal) + d) / denom
        
        # Calculate the intersection points
        intersection_points = ray_positions + t[:, np.newaxis] * ray_directions

        # Find the rays that fall out of the image plane 
        outOfBoundInd = (intersection_points[:, 0] > (self.width/2)) | \
            (intersection_points[:, 0] < (-self.width/2)) | \
            (intersection_points[:, 1] > (self.height/2)) | \
            (intersection_points[:, 1] < (-self.height/2)) 
        
        # Only replace the in bound ray positions 
        ray_positions[~outOfBoundInd] = intersection_points[~outOfBoundInd]
        
        # Update the current ray batch positions 
        og_positions[np.where(self.rayBatch.Sequential() == 1)] = ray_positions
        self.rayBatch.SetPosition(og_positions)

        # Copy the positions into path 
        self.rayPath.append(np.copy(og_positions))

        self.rayBatch.SetVignette(np.where(~valid_rays & outOfBoundInd))

        
    
    def _integralRays(self, plotResult = True):
        """
        Taking integral over the rays arriving at the image plane. 

        :param plotResult: whether to show the resulting plot or not. 
        """

        pxPitch = self.width / self.horizontalPx 
        pxOffset = np.array([self.horizontalPx/2, self.verticalPx/2, 0])

        # Find the rays that arrived at the the image plane 
        rayHitIndex = np.where(np.isclose(self.rayBatch.value[:, 2], self._zPos))

        # Translate the intersections from 3D image space to 2D pixel-based space
        rayPos = self.rayBatch.Position()[rayHitIndex] / pxPitch + pxOffset
        rayColor = self.rayBatch.Radiant()[rayHitIndex]

        # Convert ray position into pixel position 
        rayPos = np.floor(rayPos).astype(int)
        # Create pixel grid 
        radiantGrid = np.zeros( (self.horizontalPx, self.verticalPx) )

        # Sum up the radiant 
        np.add.at(radiantGrid, (rayPos[:, 0], rayPos[:, 1]), rayColor)

        if (plotResult):
            plt.imshow(radiantGrid, cmap='gray', vmin=0, vmax=np.max(radiantGrid))
            plt.colorbar()  # Optional: Add a colorbar to show intensity values
            plt.show()



    def _WaveLengthToRGB(self, wavelengths, OneValueNorm=False, Gamma=0.80, IntensityMax=255):
        """
        Convert a NumPy array of wavelengths to corresponding RGB values.
        Returns [0, 0, 0] for wavelengths outside the visible spectrum (380 nm to 780 nm).
        
        :param wavelengths: NumPy array of wavelengths.
        :param OneValueNorm: Whether to normalize the output RGB values to [0, 1].
        :param Gamma: Gamma correction value.
        :param IntensityMax: Maximum intensity value for RGB.
        :return: NumPy array of RGB values for each wavelength.
        """
        UVCut = LambdaLines["i"]      # 365 
        purpleLine = LambdaLines["h"] # 404
        blueLine = LambdaLines["g"]   # 435 
        cyanLine = LambdaLines["F"]   # 486
        greenLine = LambdaLines["e"]  # 546 
        yellowLine = LambdaLines["d"] # 587 
        orangeLine = LambdaLines["C'"]# 643 
        redLine = LambdaLines["r"]    # 706 
        IRCut = LambdaLines["A'"]     # 768 

        wavelengths = np.array(wavelengths)  # Ensure it's a NumPy array if not already

        # Initialize arrays for Red, Green, Blue, and factor
        Red = np.zeros_like(wavelengths, dtype=np.float32)
        Green = np.zeros_like(wavelengths, dtype=np.float32)
        Blue = np.zeros_like(wavelengths, dtype=np.float32)
        factor = np.zeros_like(wavelengths, dtype=np.float32)

        # Red, Green, Blue calculation based on wavelength range
        mask_Purple = (wavelengths >= UVCut) & (wavelengths < purpleLine)
        Red[mask_Purple] = -(wavelengths[mask_Purple] - purpleLine) / (purpleLine - UVCut)
        Blue[mask_Purple] = 1.0

        mask_Blue = (wavelengths >= purpleLine) & (wavelengths < blueLine)
        Green[mask_Blue] = (wavelengths[mask_Blue] - purpleLine) / (blueLine - purpleLine)
        Blue[mask_Blue] = 1.0

        mask_Cyan = (wavelengths >= blueLine) & (wavelengths < cyanLine)
        Green[mask_Cyan] = 1.0
        Blue[mask_Cyan] = -(wavelengths[mask_Cyan] - cyanLine) / (cyanLine - blueLine)

        mask_Green = (wavelengths >= cyanLine) & (wavelengths < greenLine)
        Red[mask_Green] = (wavelengths[mask_Green] - cyanLine) / (greenLine - cyanLine)
        Green[mask_Green] = 1.0

        mask_Yellow = (wavelengths >= greenLine) & (wavelengths < yellowLine)
        Red[mask_Yellow] = 1.0
        Green[mask_Yellow] = (-(wavelengths[mask_Yellow] - yellowLine) / (yellowLine - greenLine)) 

        mask_Orange = (wavelengths >= yellowLine) & (wavelengths < orangeLine)
        Red[mask_Orange] = 1.0
        Green[mask_Orange] =  ((wavelengths[mask_Orange] - orangeLine) / (orangeLine - yellowLine)) 

        mask_Red = (wavelengths >= orangeLine) & (wavelengths < redLine)
        Red[mask_Red] = 1.0 - (wavelengths[mask_Red] - redLine) / (redLine - orangeLine)

        mask_IR = (wavelengths >= redLine) 
        Red[mask_IR] = 1.0 - (wavelengths[mask_IR] - orangeLine) / (IRCut - redLine)

        # Let the intensity fall off near the vision limits
        mask_380_420 = (wavelengths >= 380) & (wavelengths < 420)
        factor[mask_380_420] = 0.3 + 0.7 * (wavelengths[mask_380_420] - 380) / (420 - 380)

        mask_420_701 = (wavelengths >= 420) & (wavelengths < 701)
        factor[mask_420_701] = 1.0

        mask_701_780 = (wavelengths >= 701) & (wavelengths < 781)
        factor[mask_701_780] = 0.3 + 0.7 * (780 - wavelengths[mask_701_780]) / (780 - 700)

        # Apply gamma correction and intensity factor
        Red = np.where(Red != 0, IntensityMax * np.power(Red * factor, Gamma), 0).astype(int)
        Green = np.where(Green != 0, IntensityMax * np.power(Green * factor, Gamma), 0).astype(int)
        Blue = np.where(Blue != 0, IntensityMax * np.power(Blue * factor, Gamma), 0).astype(int)

        # Set RGB to [0, 0, 0] for wavelengths outside the visible spectrum (380 nm to 780 nm)
        outside_visible_mask = (wavelengths < UVCut) | (wavelengths >= IRCut)
        Red[outside_visible_mask] = 0
        Green[outside_visible_mask] = 0
        Blue[outside_visible_mask] = 0

        # Stack the RGB components into an (N, 3) array where N is the number of wavelengths
        rgb = np.stack([Red, Green, Blue], axis=-1)

        # Optionally normalize RGB values to range [0, 1]
        if OneValueNorm:
            rgb = rgb / IntensityMax

        return rgb


    def _RGBToWavelength(self, RGB, 
                         primaries = {"R": "C'", "G": "e", "B":"g"}, 
                         secondaries = ["F", "D"], 
                         UVIRcut = ["i", "A'"],
                         bitDepth=8):
        """
        Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
        
        :param RGB: RGB values
        :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"})
        :param secondaries: A dictionary mapping secondary colors to wavelength lines (optional)
        :param UVIRcut: Cut wavelength for ultraviolet and infrared, the first term is UV and the second is IR. 
        :return: A NumPy array of wavelengths corresponding to the input RGB array
        """

        # Normalize RGB values to the range [0, 1]
        bits = 2.0 ** bitDepth - 1

        wavelengths = np.array([
            LambdaLines[primaries["R"]], 
            LambdaLines[primaries["G"]], 
            LambdaLines[primaries["B"]]
        ])

        radiants = np.array(RGB) / bits
        print(radiants)

        if (len(secondaries) > 0):
            for secondary in secondaries:
                currentWavelength = LambdaLines[secondary]
                currentRadiant = 0
                wavelengths = np.append(wavelengths, currentWavelength)

                # Between IR limit and Red line 
                if(currentWavelength < LambdaLines[UVIRcut[1]] and currentWavelength > LambdaLines[primaries["R"]]):
                    # Using red radiant and reduce the intensity depending on how far it is away from the red line 
                    currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[primaries["R"]]) / (LambdaLines[UVIRcut[1]] - LambdaLines[primaries["R"]]) )

                # Between Red line and Green line 
                elif(currentWavelength < LambdaLines[primaries["R"]] and currentWavelength > LambdaLines[primaries["G"]]):
                    # Find the ratio between red and green 
                    ratio = (currentWavelength - LambdaLines[primaries["G"]]) / (LambdaLines[primaries["R"]] - LambdaLines[primaries["G"]])

                    currentRadiant = radiants[0] * ratio + radiants[1] * (1 - ratio)

                # Between Green line and Blue line 
                elif(currentWavelength < LambdaLines[primaries["G"]] and currentWavelength > LambdaLines[primaries["B"]]):
                    # Find the ratio between green and blue 
                    ratio = (currentWavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]] - LambdaLines[primaries["B"]])

                    currentRadiant = radiants[1] * ratio + radiants[2] * (1 - ratio)

                # Between Blue line and UV limit 
                elif(currentWavelength < LambdaLines[primaries["B"]] and currentWavelength > LambdaLines[UVIRcut[0]]):
                    currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]]) )

                radiants = np.append(radiants, currentRadiant)

        return (wavelengths, radiants)
    

    def _WavelengthToRGB(self):
        # Should there be a separate conversion function? 
        pass 


def main():
    imager = Imager() 
    imager.Test()


if __name__ == "__main__":
    main()