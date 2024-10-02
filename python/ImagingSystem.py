
from PIL import Image

from Lens import * 

class ImagingSystem:
    def __init__(self):
        self.lens = None 
        self.imager = None 
        self.rayBatch = None 
        self.point = None 
        self.inputImage = None 

    
    def AddLens(self, lens):
        self.lens = lens 

    def SinglePointSpot(self, pointPosition):
        pass 


    def _integralRays(self, surfaceIndex):
        """
        Taking integral over the rays arriving at the image plane. 

        :param surfaceIndex: the index of the surface to intersect. 
        """
        if(not self.surfaces[surfaceIndex].IsImagePlane()):
           # TODO: add spherical imager? 
           return 
        
        cumulativeThickness = self.surfaces[surfaceIndex].cumulativeThickness
        ipSize = self.surfaces[surfaceIndex].ImagePlaneSize()
        pxDimension = self.surfaces[surfaceIndex].ImagePlanePx()

        pxPitch = ipSize[0] / pxDimension[0] 
        pxOffset = np.array([pxDimension[0]/2, pxDimension[1]/2, 0])

        # Find the rays that arrived at the the image plane 
        rayHitIndex = np.where(np.isclose(self.rayBatch.value[:, 2], cumulativeThickness))

        # Translate the intersections from 3D image space to 2D pixel-based space
        rayPos = self.rayBatch.Position()[rayHitIndex] / pxPitch + pxOffset
        rayColor = self.rayBatch.Radiant()[rayHitIndex]

        # Convert ray position into pixel position 
        rayPos = np.floor(rayPos).astype(int)
        # Create pixel grid 
        radiantGrid = np.zeros( (pxDimension[0], pxDimension[1]) )

        # Sum up the radiant 
        np.add.at(radiantGrid, (rayPos[:, 0], rayPos[:, 1]), rayColor)

        # Register the radiant grid to the spot 
        self.spot = radiantGrid
        plt.imshow(radiantGrid, cmap='gray', vmin=0, vmax=np.max(radiantGrid))
        plt.colorbar()  # Optional: Add a colorbar to show intensity values
        plt.show()



def main():
    biotar = Lens() 

    # Zeiss Biotar 50 1.4 
    biotar.AddSurfacve(Surface(41.8,   5.375, 17, "BAF9"))
    biotar.AddSurfacve(Surface(160.5,  0.825, 17))
    biotar.AddSurfacve(Surface(22.4,	7.775, 16, "SK10"))
    biotar.AddSurfacve(Surface(-575,	2.525, 16, "LZ_LF5"))
    biotar.AddSurfacve(Surface(14.15,	9.45, 11))
    biotar.AddSurfacve(Surface(-19.25,	2.525, 11, "SF5"))
    biotar.AddSurfacve(Surface(25.25,	10.61, 13, "BAF9"))
    biotar.AddSurfacve(Surface(-26.6,	0.485, 13))
    biotar.AddSurfacve(Surface(53, 	6.95, 14, "BAF9"))
    biotar.AddSurfacve(Surface(-60,	32.3552, 14))


    imgSys = ImagingSystem() 
    imgSys.AddLens(biotar)
    imgSys.SinglePointSpot(np.array([150, 100, -500]))



if __name__ == "__main__":
    main()