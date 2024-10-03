
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

    def IntegralRays(self, raybatch):
        self.rayBatch = raybatch
        self._ImagePlaneIntersections() 
        self._integralRays() 

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
        combinedValidInd = ~outOfBoundInd & valid_rays
        
        # Only replace the in bound ray positions 
        ray_positions[~outOfBoundInd] = intersection_points[~outOfBoundInd]
        
        # Update the current ray batch positions 
        og_positions[np.where(self.rayBatch.Sequential() == 1)] = ray_positions
        self.rayBatch.SetPosition(og_positions)

        self.rayBatch.SetVignette(np.where(~valid_rays & outOfBoundInd))
        

    def _integralRays(self):
        """
        Taking integral over the rays arriving at the image plane. 

        :param surfaceIndex: the index of the surface to intersect. 
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

        # Register the radiant grid to the spot 
        self.spot = radiantGrid
        plt.imshow(radiantGrid, cmap='gray', vmin=0, vmax=np.max(radiantGrid))
        plt.colorbar()  # Optional: Add a colorbar to show intensity values
        plt.show()

