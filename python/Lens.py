

import PlotTest
from Surface import *
from Util import * 
from RayBatch import * 

import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Lens:
    def __init__(self):
        self.elements = []
        self.env = "AIR" # The environment it is submerged in, air by default 
        
        self.rayBatch = RayBatch([])
        self.rayPath = [] # Rays with only position info on each surface 

        self.emitters = []
        self.emitter = None 


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters 
        """
        
        currentT = 0

        for e in self.elements:
            e.SetCumulative(currentT)
            currentT += e.thickness


    def AddSurfacve(self, surface, insertAfter = None):
        self.elements.append(surface)


    def SinglePointSpot(self, posP):
        """
        Place a single point source and accquire its spot. 
        """
        self._initRays(posP)
        self._propograte() 
        

    def DrawLens(self, drawSrufaces = True, drawRays = False):

        ax = PlotTest.Setup3Dplot()
        PlotTest.SetUnifScale(ax)

        # Draw every surfaces 
        for l in self.elements:
            PlotTest.DrawSpherical(ax, l.radius, l.clearSemiDiameter, l.cumulativeThickness)

        # Draw the path of rays 
        if (drawRays):
            for i in range(len(self.rayPath) - 1):
                for v1, v2 in zip(self.rayPath[i], self.rayPath[i+1]):
                    PlotTest.DrawLine(ax, v1, v2, lineColor = "r", lineWidth = 0.5) 
        
        plt.show()


    def Test(self):

        
        self.rayBatch = RayBatch(np.array([
            [0, 0, 3, 0, 0, 1, 550, 1, 0, 1],
            [0, 1, 2, 0, 0.3, 0.9, 470, 1, 1, 1],
            [0, 2, 1, 0, 0.8, 0.3, 640, 1, 2, 0],
            [0, 3, 0, 0, 0.6, 0.6, 580, 1, 3, 0],
        ]))
        print(self.rayBatch.SurfaceIndex())

        
    # ============================================================================
    """ ====================================================================== """
    # ============================================================================

    def _findB(self, posA, posC, posP):
        """
        Find the position of point B. 
        :param posA: position of point A.
        :param posC: position of point C.
        :param posP: position of point P.
        """
        vecPA = posA - posP   
        vecPC = posC - posP  
        vecN = Normalized((Normalized(vecPA) + Normalized(vecPC)) / 2)
        
        vecPCN = Normalized(vecPC)
        t = (np.dot(vecN, (posA - posC))) / np.dot(vecN, vecPCN)

        return posC + vecPCN * t 

    def _ellipsePeripheral(self, posA, posB, posC, posP, d, r, useDistribution = True):
        """
        Find the points on the ellipse and align it to the AB plane. 

        :param posA: position of point A. 
        :param posB: position of point B. 
        :param posC: position of point C. 
        :param posP: position of point P. 
        :param d: clear semi diameter of the surface. 
        :param r: radius of the surface. 
        :param useDistribution: when enabled, the method returns a distribution of points in the ellipse area instead of the points representing the outline of the ellipse. 
        """
        offset = np.array([0, 0, r - np.sqrt(r**2 - d**2)])

        # Util vectors 
        P_xy_projection = Normalized(np.array([posP[0], posP[1], 0]))

        # On axis scenario 
        if (np.linalg.norm(P_xy_projection) == 0):
            P_xy_projection = np.array([d, 0, 0])

        vecCA = posA - posC
        
        # Lengths to calculate semi-major axis length 
        BB = abs((2 * d) * ((posP[2] - posB[2]) / posP[2]))
        AC = np.linalg.norm(posA - posC)
        
        # Semi-major axis length 
        a = np.linalg.norm(posA - posB) / 2
        b = np.sqrt(BB * AC) / 2
        
        # Calculate the ellipse 
        if (useDistribution):
            points = CircularDistribution()
            points = np.transpose(np.transpose(points) * np.array([b, a, 0]))
        else:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = b * np.cos(theta) 
            y = a * np.sin(theta) 
            z = np.ones(len(x)) * posA[2]
            points = np.array([x, y, z])

        # Rotate the ellipse to it faces the right direction in the world xy plane,
        # i.e., one of its axis coincides with the tangential plane 
        theta_1 = angleBetweenVectors(posA, np.array([0, 1, 0]))
        trans_1 = Rotation(-theta_1, np.array([0, 0, 1]), points)
        
        # Move the points to be in tangent with A 
        trans_1 = Translate(trans_1, P_xy_projection * (d - a)) 
        
        # Rotate the ellipse around A it fits into the AB plane 
        theta = angleBetweenVectors(posB-posA, posC-posA)
        axis = Normalized(np.array([-vecCA[1], vecCA[0], 0]))
        trans_2 = Translate(trans_1, -posA)
        trans_2 = Rotation(-theta, axis, trans_2)
        trans_2 = Translate(trans_2, posA)
        
        return trans_2

    def _sphericalIntersections(self, surfaceIndex):
        """
        This method update the position of rays as they hit the indexed surface. 
        """
        
        # Set up parameters to use later 
        ray_origins = self.rayBatch.Position()[np.where(self.rayBatch.Sequential() == 1)]
        ray_directions = self.rayBatch.Direction()[np.where(self.rayBatch.Sequential() == 1)]
        sphere_center = self.elements[surfaceIndex].radiusCenter
        sphere_radius = self.elements[surfaceIndex].radius
        clear_semi_diameter = self.elements[surfaceIndex].clearSemiDiameter
        cumulative_thickness = self.elements[surfaceIndex].cumulativeThickness
        
        # Normalize the direction vector
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        # Coefficients for the quadratic equation
        oc = ray_origins - sphere_center
        A = np.sum(ray_directions**2, axis=1)  # A = dot(ray_direction, ray_direction)
        B = 2.0 * np.sum(oc * ray_directions, axis=1)
        C = np.sum(oc**2, axis=1) - sphere_radius**2

        # Discriminant of the quadratic equation
        discriminant = B**2 - 4*A*C

        # Set the non-intersect and tangent rays to be vignetted 
        vignettedIndices = np.where(discriminant <= 0)
        self.rayBatch.SetVignette(vignettedIndices)


        neg_limit = cumulative_thickness
        difference = (abs(sphere_radius) - np.sqrt(sphere_radius**2 - clear_semi_diameter**2))
        if (sphere_radius > 0):
            pos_limit = cumulative_thickness + difference
        else:
            pos_limit = cumulative_thickness - difference
            pos_limit, neg_limit = neg_limit, pos_limit 

        invertVignette = np.where(discriminant > 0)
        sqrt_discriminant = np.sqrt(discriminant[invertVignette])
        
        # Calculate the intersection points
        if(sphere_radius > 0):
            t1 = (-B[invertVignette] - sqrt_discriminant) / (2*A[invertVignette])
            intersection_points = ray_origins[invertVignette] + t1[:, np.newaxis] * ray_directions[invertVignette]
        else: 
            t2 = (-B[invertVignette] + sqrt_discriminant) / (2*A[invertVignette])
            intersection_points = ray_origins[invertVignette] + t2[:, np.newaxis] * ray_directions[invertVignette]
        
        ray_origins[invertVignette] = intersection_points
        self.rayBatch.SetPosition(ray_origins)
        self.rayPath.append(self.rayBatch.Position())


        
        

    def _initRays(self, posP, wavelength = 550):
        r = self.elements[0].radius
        sd = self.elements[0].clearSemiDiameter

        P_xy_projection = np.array([posP[0], posP[1], 0])
        offset = np.array([0, 0, abs(r) - np.sqrt(r**2 - sd**2)])
        posA = sd * ( P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
        posC = sd * (-P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
        posB = self._findB(posA, posC, posP)

        points = self._ellipsePeripheral(posA, posB, posC, posP, sd, r) # Sample points in the ellipse area 

        vecs = Normalized(np.transpose(points) - posP)

        # Spawn rays with position of P to be the position for each ray 
        self.rayBatch = RayBatch(
            np.tile(np.array([posP[0], posP[1], posP[2],
                               0, 0, 0, 
                               wavelength, 1, 0, 1]), 
                    (vecs.shape[0], 1))
        )

        # Set the initial rays' direction into the newly spawned rays 
        self.rayBatch.SetDirection(vecs)

        # Append the starting point into ray path 
        self.rayPath.append(
            np.tile(np.array([posP[0], posP[1], posP[2]]), (vecs.shape[0], 1))
            )


    def _propograte(self):

        self._sphericalIntersections(0)
        # for i in range(len(self.elements)):
        #     self._sphericalIntersections(i)

    def _initSinglePoint(self):
        pass 



def main():
    singlet = Lens() 
    singlet.AddSurfacve(Surface(-20, 4, 6, "LAF8"))
    singlet.AddSurfacve(Surface(-10, 4, 6.6))
    singlet.UpdateLens()

    singlet.SinglePointSpot(np.array([1, 0.5, -10]))

    singlet.DrawLens(drawRays=True)

    


if __name__ == "__main__":
    main()