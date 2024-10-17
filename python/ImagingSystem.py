
from PIL import Image

from Lens import * 
from Imager import * 
from Util import * 

class ImagingSystem:
    def __init__(self):

        self.lens = None 
        self.imager = None 
        self.rayBatch = None 

        self.rayPath = None 

        self.point = None 
        self.inputImage = None 

    
    def AddLens(self, lens):
        self.lens = lens 

    def AddImager(self, imager):
        self.imager = imager 

    def SinglePointSpot(self, pointPosition):
        self._initRays(pointPosition)
        self.lens.SetIncomingRayBatch(self.rayBatch)
        self.rayBatch = self.lens.Propagate() 
        self.imager.IntegralRays(self.rayBatch)

        self.rayPath = self.lens.rayPath


    def DrawSystem(self, drawSurfaces=True, drawPath=True, rayPathMax=32):
        """
        This method draws the optical system in 3D for easier inspection. 

        :param drawSurfaces: when enabled, surfaces will be drawn. 
        :param drawPath: when enabled, ray paths will be drawn.
        :param rayPathMax: max number of ray paths to draw, since there might be millions of ray paths recorded. 
        """

        ax = PlotTest.Setup3Dplot()
        PlotTest.SetUnifScale(ax, self.lens.totalLength)

        if(drawSurfaces):
            self.lens.DrawLens(ax = ax)

            ipSize = np.array([self.imager.width, self.imager.height]) / 2
            zPos = self.imager.GetZPos()
            corners = np.array([
                [ipSize[0], ipSize[1], zPos],
                [-ipSize[0], ipSize[1], zPos],
                [-ipSize[0], -ipSize[1], zPos],
                [ipSize[0], -ipSize[1], zPos]
            ])
            vertices = [[corners[0], corners[1], corners[2], corners[3]]]
            ax.add_collection3d(Poly3DCollection(vertices, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.5, zorder=0))

        if(drawPath):

            # Accquire the raypath 
            rayPath = self.lens.rayPath 
            rayPath.extend(self.imager.rayPath)

            # prune the paths to avoid drawing too many 
            deletionIndices = np.random.choice(len(rayPath[0]), rayPathMax, replace=False)
            prunedPath = [arr[deletionIndices] for arr in rayPath]

            rayThickness = 0.25 
            
            # Iterate from surface to surface and draw the lines
            for i in range(len(prunedPath) - 1):
                for v1, v2 in zip(prunedPath[i], prunedPath[i+1]):
                    PlotTest.DrawLine(ax, v1, v2, lineColor = "r", lineWidth = rayThickness, zorder=10) 
            

        plt.show()


    # ==================================================================
    """ ============================================================ """
    # ==================================================================

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


    def _ellipsePeripheral(self, posA, posB, posC, posP, sd, r, samplePoints = 100, useDistribution = True):
        """
        Find the points on the ellipse perpendicular to the incident direction that will be used as initial ray cast points. 

        :param posA: position of point A. 
        :param posB: position of point B. 
        :param posC: position of point C. 
        :param posP: position of point P. 
        :param sd: clear semi diameter of the surface. 
        :param r: radius of the surface. 
        :param useDistribution: when enabled, the method returns a distribution of points in the ellipse area instead of the points representing the outline of the ellipse. 
        """
        offset = np.array([0, 0, posA[2]])
        onAxis = False 

        # On axis scenario 
        if (np.isclose(posP[0], 0) and np.isclose(posP[1], 0)):
            P_xy_projection = np.array([sd, 0, 0])
            onAxis = True 
        else:
            # Util vectors 
            P_xy_projection = Normalized(np.array([posP[0], posP[1], 0]))

        vecCA = posA - posC
        
        # On axis rays can be grealty simplified 
        if(onAxis):
            # Generate the internal samples  
            if (useDistribution):
                # Move the point along the z axis 
                points = np.transpose(RandomEllipticalDistribution(samplePoints=samplePoints)) + offset
                # Scale it on the two semi-major axis 
                points = np.transpose(points * np.array([sd, sd, 1]))
            
            # Generate the contour of the ellipse 
            else:
                theta = np.linspace(0, 2 * np.pi, 100)
                x = np.cos(theta) 
                y = np.sin(theta) 
                z = np.ones(len(x)) * posA[2]
                points = np.array([x, y, z])
            return points 
        
        # Off axis rays 
        else:
            # Lengths to calculate semi-major axis length 
            BB = abs((2 * sd) * ((posP[2] - posB[2]) / posP[2]))
            AC = np.linalg.norm(posA - posC)
            
            # Semi-major axis length 
            a = np.linalg.norm(posA - posB) / 2
            b = np.sqrt(BB * AC) / 2

            # Calculate the ellipse 
            if (useDistribution):
                # Move the point along the z axis 
                points = np.transpose(RandomEllipticalDistribution(samplePoints=samplePoints)) + offset
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
            trans_1 = Translate(trans_1, P_xy_projection * (sd - a)) 
            
            # Rotate the ellipse around A it fits into the AB plane 
            theta = angleBetweenVectors(posB-posA, posC-posA)
            axis = Normalized(np.array([-vecCA[1], vecCA[0], 0]))
            trans_2 = Translate(trans_1, -posA)
            trans_2 = Rotation(-theta, axis, trans_2)
            trans_2 = Translate(trans_2, posA)

            return trans_2
        

    def _singlePointRaybatch(self, posP, RGB=[255, 128, 1], bitDepth=8):
        wavelengths, radiants = RGBToWavelength(RGB)

        wavelength = 550 
        
        firstSurface = self.lens.surfaces[0] 
        r = firstSurface.radius
        sd = firstSurface.clearSemiDiameter

        P_xy_projection = np.array([posP[0], posP[1], 0])
        offset = np.array([0, 0, abs(r) - np.sqrt(r**2 - sd**2)]) * np.sign(r)
        if(not np.isclose(np.linalg.norm(P_xy_projection), 0)):
            posA = sd * ( P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
            posC = sd * (-P_xy_projection / np.linalg.norm(P_xy_projection) ) + offset
        else:
            posA = np.array([sd, 0, 0]) + offset
            posC = np.array([sd, 0, 0]) + offset
        posB = self._findB(posA, posC, posP)

        points = np.transpose(self._ellipsePeripheral(posA, posB, posC, posP, sd, r)) # Sample points in the ellipse area 
        self._temp = self._ellipsePeripheral(posA, posB, posC, posP, sd, r, False) # Points that form the edge of the ellipse 

        vecs = ArrayNormalized(points - posP)

        # Create the ray batch 
        # For some reason vecs is often not registered with indexing assignment, the hstack is thus used to force compose the raybatch. 
        mat1 = np.tile(np.array([posP[0], posP[1], posP[2]]), (vecs.shape[0], 1))
        mat2 = np.tile(np.array([wavelength, 1, 0, 1]), (vecs.shape[0], 1))
        mat = np.hstack((np.hstack((mat1, vecs)), mat2))
        
        return np.hstack((np.hstack((mat1, vecs)), mat2))


    def _initRays(self, posP):
        mat = self._singlePointRaybatch(posP)

        self.rayBatch = RayBatch( mat )



    
def main():
    biotar = Lens() 

    # Set up the lens 
    # Zeiss Biotar 50 1.4 
    biotar.AddSurfacve(Surface(41.8,    5.375,  17, "BAF9"))
    biotar.AddSurfacve(Surface(160.5,   0.825,  17))
    biotar.AddSurfacve(Surface(22.4,	7.775,  16, "SK10"))
    biotar.AddSurfacve(Surface(-575,	2.525,  16, "LZ_LF5"))
    biotar.AddSurfacve(Surface(14.15,	9.45,   11))
    biotar.AddSurfacve(Surface(-19.25,	2.525,  11, "SF5"))
    biotar.AddSurfacve(Surface(25.25,	10.61,  13, "BAF9"))
    biotar.AddSurfacve(Surface(-26.6,	0.485,  13))
    biotar.AddSurfacve(Surface(53, 	    6.95,   14, "BAF9"))
    biotar.AddSurfacve(Surface(-60,	    32.3552, 14))
    # Update immediately after all the surfaces are created 
    biotar.UpdateLens() 

    # Set up the imager 
    imager = Imager(bfd=32.3552)

    # Assemble the imaging system 
    imgSys = ImagingSystem() 
    imgSys.AddLens(biotar)
    imgSys.AddImager(imager)
    imgSys.imager.SetLensLength(imgSys.lens.totalLength)

    # Propagate light 
    imgSys.SinglePointSpot(np.array([150, 100, -500]))

    imgSys.DrawSystem()
    



if __name__ == "__main__":
    main()