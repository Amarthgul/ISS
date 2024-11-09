
from PIL import Image
import gc

from Lens import * 
from Imager import * 
from Util import * 
from ObjectSpace import * 


# Random generator from the Util module
# This uses the same seed so that the result is deterministic  
RNG = rng 


class ImagingSystem:
    def __init__(self):

        self.lens = None 
        self.imager = None 
        self.rayBatch = None 

        self.rayPath = None 

        # Color-wavelength conversion 
        self.primaries = {"R": "C'", "G": "e", "B":"g"}
        self.secondaries = []#["F", "D"]
        self.UVIRcut = ["i", "A'"]

        self.point = None 
        self.inputImage = None 

        self._perSpotMaxSample = 100

    
    def AddLens(self, lens):
        self.lens = lens 

    def AddImager(self, imager):
        self.imager = imager 

    def SinglePointSpot(self, point):
        mat = self._singlePointRaybatch(point.GetPosition(), 
                                        point.GetColorRGB(), 
                                        point.GetBitDepth())
        self.rayBatch = RayBatch( mat )
        self.lens.SetIncomingRayBatch(self.rayBatch)
        self.rayBatch = self.lens.Propagate() 

        self.imager.IntegralRays(self.rayBatch,
            self.primaries, self.secondaries, self.UVIRcut)

        self.rayPath = self.lens.rayPath


    def Image2DPropagation(self, image):
        print(image.sampleX, "     ",  image.sampleY)
        # Using the python list and append is faster than numpy vstack
        mat = []  # possibily due to memory allocation 

        start = time.time()
        sX = image.sampleX
        sY = image.sampleY

        for i in range(sX- 1):
            print("At {:.2f} percent".format(100 * i/sX))
            for j in range(sY - 1):
                pointdata = image.pointData[j, i]
                temp = self._singlePointRaybatch(pointdata[:3], 
                                                pointdata[3:6], 
                                                image.bitDepth)
                if(len(temp) > 0): # Only append for non zero array 
                    mat.append(temp)
        end = time.time()
        if(DEVELOPER_MODE):
            print("Mat creation time: ", end - start)

        mat = np.vstack(mat)
        print("stacked with shape  ", mat.shape)

        self.rayBatch = RayBatch( mat )
        self.lens.SetIncomingRayBatch(self.rayBatch)
        self.rayBatch = self.lens.Propagate() 

        self.imager.IntegralRays(self.rayBatch,
            self.primaries, self.secondaries, self.UVIRcut)

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
            deletionIndices = RNG.choice(len(rayPath[0]), rayPathMax, replace=False)
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


    def _ellipsePeripheral(self, posA, posB, posC, posP, sd, r, samplePoints = SOME_BIG_CONST, useDistribution = True):
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
        

    def _singlePointRaybatch(self, posP, RGB=[1, 1, 1], bitDepth=8):
        """
        Generate the initial rayBatch for a single point light source. 
        """
        lumi = LumiPeak(RGB)
        if(lumi == 0): return [] 

        # Accquire all wavelengths and corresponding radiants  
        wavelengths, radiants = RGBToWavelength(RGB, self.primaries, self.secondaries, self.UVIRcut)
        
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

        sampleAmount = int(lumi * self._perSpotMaxSample)
        points = np.transpose(self._ellipsePeripheral(posA, posB, posC, posP, sd, r, sampleAmount)) # Sample points in the ellipse area 
        vecs = ArrayNormalized(points - posP)

        # Radiant is used to calculate the amount of rays 
        sampleCount = len(points)
        wavelengthCount = (Partition(radiants) * sampleCount).astype(int)

        # Due to integer floor/ceiling, there might be less entry than the count. The missing ones are randomly added. 
        while(np.sum(wavelengthCount) != sampleCount):
            wavelengthCount[RNG.integers(len(wavelengthCount)-1)] += 1
        wavelengthArray = np.repeat(wavelengths, wavelengthCount)

        # Create the ray batch 
        # For some reason vecs is often not registered with indexing assignment, the hstack is thus used to force compose the raybatch. 
        mat1 = np.tile(np.array([posP[0], posP[1], posP[2]]), (vecs.shape[0], 1))
        mat2 = np.tile(np.array([1, 0, 1]), (vecs.shape[0], 1))
        #mat = np.hstack((np.hstack((mat1, vecs)), mat2))
        
        return np.hstack((mat1, vecs, np.transpose(wavelengthArray)[:, np.newaxis], mat2))




    
def main():
    biotar = Lens() 

    # Set up the lens 
    # Zeiss Biotar 50 1.4 
    biotar.AddSurface(Surface(41.8,     5.375,  17, "BAF9"))
    biotar.AddSurface(Surface(160.5,    0.825,  17))
    biotar.AddSurface(Surface(22.4,	    7.775,  16, "SK10"))
    biotar.AddSurface(Surface(-575,	    2.525,  16, "LZ_LF5"))
    biotar.AddSurface(Surface(14.15,	9.45,   11))
    biotar.AddSurface(Surface(-19.25,	2.525,  11, "SF5"))
    biotar.AddSurface(Surface(25.25,	10.61,  13, "BAF9"))
    biotar.AddSurface(Surface(-26.6,	0.485,  13))
    biotar.AddSurface(Surface(53, 	    6.95,   14, "BAF9"))
    biotar.AddSurface(Surface(-60,	    32.3552, 14))
    # Update immediately after all the surfaces are defined  
    biotar.UpdateLens() 

    # Set up the imager 32.3552 (35 for 1500 distance)
    imager = Imager(bfd=35)

    # Assemble the imaging system 
    imgSys = ImagingSystem() 
    imgSys.AddLens(biotar)
    imgSys.AddImager(imager)
    imgSys.imager.SetLensLength(imgSys.lens.totalLength)

    # Create objects 
    # testPoint = Point()
    # testPoint.fieldX = 15
    # testPoint.fieldY = 10
    # testPoint.distance = 700
    # testPoint.RGB = np.array([0.4, 0.7, 0.1])
    # imgSys.SinglePointSpot(testPoint)

    # Create 2D image in object space 
    testImgPath = r"resources/ISO12233.jpg"
    testImg = Image2D(testImgPath)
    imgSys.Image2DPropagation(testImg)

    #imgSys.DrawSystem()
    



if __name__ == "__main__":
    main()