

import PlotTest
from Surface import *
from Util import * 
from RayBatch import * 
from Material import * 
import time

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

        self._envMaterial = None 
        self._temp = None # Variable not to be taken serieously 


    def UpdateLens(self):
        """
        Iterate throught the elements and update their relative parameters 
        """
        
        currentT = 0

        for e in self.elements:
            e.SetCumulative(currentT)
            e.SetFrontVertex(np.array([0, 0, currentT]))
            currentT += e.thickness

        self._envMaterial = Material(self.env)


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

        PlotTest.Draw3D(ax, self._temp[0], self._temp[1], self._temp[2])
        # Draw every surfaces 
        for l in self.elements:
            PlotTest.DrawSpherical(ax, l.radius, l.clearSemiDiameter, l.cumulativeThickness)

        # Draw the path of rays 
        if (drawRays):
            for i in range(len(self.rayPath) - 1):
                for v1, v2 in zip(self.rayPath[i], self.rayPath[i+1]):
                    PlotTest.DrawLine(ax, v1, v2, lineColor = "r", lineWidth = 0.5) 
        
        lastSurfacePos = self.rayBatch.Position()
        for v1, v2 in zip(lastSurfacePos, lastSurfacePos+self.rayBatch.Direction()*5):
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
        offset = np.array([0, 0, posA[2]])

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
            # Move the point along the z axis 
            points = np.transpose(CircularDistribution()) + offset
            # Scale it on the two semi-major axis 
            points = np.transpose(points * np.array([b, a, 1]))
            
        else:
            # Generate the contour of the ellipse 
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
        sphere_center = self.elements[surfaceIndex].radiusCenter
        sphere_radius = self.elements[surfaceIndex].radius
        clear_semi_diameter = self.elements[surfaceIndex].clearSemiDiameter
        cumulative_thickness = self.elements[surfaceIndex].cumulativeThickness

        # Accquire only the ones that are not vignetted already 
        ray_origins = self.rayBatch.Position()[np.where(self.rayBatch.Sequential() == 1)]
        ray_directions = self.rayBatch.Direction()[np.where(self.rayBatch.Sequential() == 1)]

        # Normalize the direction vector
        ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1)[:, np.newaxis]
        # Coefficients for the quadratic equation
        oc = ray_origins - sphere_center
        A = np.sum(ray_directions**2, axis=1)  # A = dot(ray_direction, ray_direction)
        B = 2.0 * np.sum(oc * ray_directions, axis=1)
        C = np.sum(oc**2, axis=1) - sphere_radius**2

        # Discriminant of the quadratic equation
        discriminant = B**2 - 4*A*C

        # Set the intersect rays' mask array for sqrt (to avoid negative sqrt)
        interset = discriminant > 0
        intersectsIndices = np.where(interset)
        sqrt_discriminant = np.sqrt(discriminant[intersectsIndices])
        
        # Calculate the intersection points
        t = (-B[intersectsIndices] - np.sign(sphere_radius) * sqrt_discriminant) / (2*A[intersectsIndices])
        intersection_points = ray_origins[intersectsIndices] + t[:, np.newaxis] * ray_directions[intersectsIndices]

        # Find the non-intersect and tangent rays
        nonIntersect = discriminant <= 0
        # Find the rays whose intersection is outside of the clear semi diameter 
        vignetted = np.sqrt(intersection_points[:, 0]**2 + intersection_points[:, 1]**2) > clear_semi_diameter
        # Set them as vignetted 
        self.rayBatch.SetVignette(np.where(nonIntersect & vignetted))

        # Find the rays that's within clear semi diameter
        inBound = np.sqrt(intersection_points[:, 0]**2 + intersection_points[:, 1]**2) < clear_semi_diameter
        sequential = np.where(inBound & interset)
        # Put the interection points into rays 
        ray_origins[sequential] = intersection_points[sequential]

        # Update the current ray batch positions 
        self.rayBatch.SetPosition(ray_origins)

        # Copy the positions into path 
        self.rayPath.append(ray_origins)


    def _vectorsRefraction(self, surfaceIndex):
        """
        Calculates the refracted vectors given incident vectors, normal vectors, and the refractive indices.
        
        :param incident_vectors: Array of incident vectors (shape: Nx3).
        :param normal_vectors: Array of spherical surface normal vectors (shape: Nx3).
        :param n1: Refractive index of the first medium.
        :param n2: Refractive index of the second medium.
        :return: Array of refracted vectors or None if total internal reflection occurs.
        """
        # Accquire and ormalize incident and normal vectors
        incident_vectors = self.rayBatch.Direction()[np.where(self.rayBatch.Sequential() == 1)] 
        incident_vectors /= np.linalg.norm(incident_vectors, axis=1, keepdims=True)

        normal_vectors = SphericalNormal(
            self.elements[surfaceIndex].radius, 
            self.rayBatch.Position()[np.where(self.rayBatch.Sequential() == 1)], 
            self.elements[surfaceIndex].frontVertex
        ) 
        normal_vectors /= np.linalg.norm(normal_vectors, axis=1, keepdims=True)

        
        if(surfaceIndex == 0):
            n1 = self._envMaterial.RI(self.rayBatch.Wavelength(True))
        else:
            n1 = self.elements[surfaceIndex - 1].material.RI(self.rayBatch.Wavelength(True))
        n2 = self.elements[surfaceIndex].material.RI(self.rayBatch.Wavelength(True))

        # Compute the ratio of refractive indices
        n_ratio = n1 / n2
        
        # Dot product of incident vectors and normal vectors
        cos_theta_i = -np.einsum('ij,ij->i', incident_vectors, normal_vectors)
        
        # Calculate the discriminant to check for total internal reflection
        discriminant = 1 - (n_ratio ** 2) * (1 - cos_theta_i ** 2)
        
        # Handle total internal reflection (discriminant < 0)
        TIR = discriminant < 0
        refraction = discriminant >= 0
        
        # Calculate the refracted vectors, note that they may contain TIR 
        refracted_vectors = n_ratio * incident_vectors + (n_ratio * cos_theta_i - np.sqrt(discriminant))[:, np.newaxis] * normal_vectors

        # Copy the incident vectors 
        result = incident_vectors
        # Replace only the rays that are properly refracted 
        result[np.where(refraction)] = refracted_vectors[np.where(refraction)]
        ray_directions = self.rayBatch.Direction()
        ray_directions[np.where(self.rayBatch.Sequential() == 1)] = result
        self.rayBatch.SetDirection(ray_directions)
        
        self.rayBatch.SetVignette(TIR)


    def _initRays(self, posP, wavelength = 550):
        r = self.elements[0].radius
        sd = self.elements[0].clearSemiDiameter

        P_xy_projection = np.array([posP[0], posP[1], 0])
        offset = np.array([0, 0, abs(r) - np.sqrt(r**2 - sd**2)]) * np.sign(r)
        posA = sd * ( P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
        posC = sd * (-P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
        posB = self._findB(posA, posC, posP)

        points = self._ellipsePeripheral(posA, posB, posC, posP, sd, r) # Sample points in the ellipse area 
        self._temp = self._ellipsePeripheral(posA, posB, posC, posP, sd, r, False) # Points that form the edge of the ellipse 

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

        start = time.time()
        self._sphericalIntersections(0)
        self._vectorsRefraction(0)
        self._sphericalIntersections(1)
        self._vectorsRefraction(1)
        end = time.time()
        print("It took", (end - start), "seconds!")
        # for i in range(len(self.elements)):
        #     self._sphericalIntersections(i)

    def _initSinglePoint(self):
        pass 



def main():
    singlet = Lens() 
    singlet.AddSurfacve(Surface(20, 4, 6, "BAF1"))
    singlet.AddSurfacve(Surface(-10, 4, 6.6))
    singlet.UpdateLens()

    singlet.SinglePointSpot(np.array([4, 0.5, -10]))

    singlet.DrawLens(drawRays=True)

    


if __name__ == "__main__":
    main()