

import PIL.Image


from Points import PointsSource
from Util.Backend import backend as bd
from Util.Globals import ONE, TWO, INIT_PHASE_DIFF, INFINITY, FAR_DISTANCE, PRECISION_TYPE
from Raytracing.RayBatch import RayBatch



class Image2D:
    def __init__(self):
        """RGB array directly decoded from the file represneting the image"""
        self.image = None 


        """Original image file"""
        self._file = None 


        """Point source object built from the image"""
        self.pointSource = None


        """Unsigned unit in mm. If anchors are not explicitly stated, assume image at infinity"""
        self.distance = INFINITY

        """Unsigned unit in degree. If anchors are not explicitly stated, assume image in 3D fills a horizontal angle of view. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40


        """4 points data in Vec3. The 4 anchor points that pins the image in 3D space """
        self.pointAnchor = None 
        

        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


    def LoadFrom8bit(self, imgPath):
        """
        For common 8 bit image formats like jpg, bmp, and png. If a png is not 8 bit, do not use this method. Find the right bit depth method instead. 
        """

        # Read and save the original 
        self._file = PIL.Image.open(imgPath).convert("RGB")

        # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._file.height * (self.imageDimensionOverride / self._file.width))
            imageFile = self._file.resize((self.imageDimensionOverride, newHeight))
        else:
            imageFile = self._file

        # Convert into array format 
        self.image = bd.array(imageFile)

        # Normalize into [0, 1 range], this is where 8 bit kicks in 
        self.image = self.image.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        self._GeneratePointSources()


    def EmitSamplesToward(self, targets, sampleCount=64):
        pass 


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _GeneratePointSources(self):
        """
        Using the RGB and position data, generate a point source object that cooresponds to all the pixel/samples from the image input. 
        """

        # Infinty is not really workable, replace it with an approximation
        if (self.distance is INFINITY):
            zDist = -FAR_DISTANCE
        else:
            zDist = -self.distance

        # Create the 4 anchor points if they are not defined 
        if(self.pointAnchor is None):
            rad = bd.deg2rad(self.horizontalAoV) / 2
            halfX = bd.tan(rad) * zDist
            halfY = halfX * (self._file.height / self._file.width)

            self.pointAnchor = bd.array([
                [ halfX,  halfY, zDist], 
                [-halfX,  halfY, zDist], 
                [-halfX, -halfY, zDist], 
                [ halfX, -halfY, zDist], 
            ])

        sampleX = self.image.shape[1]
        sampleY = self.image.shape[0]

        u = bd.linspace(0, 1, sampleX)  # Interpolation values in x-direction
        v = bd.linspace(0, 1, sampleY)  # Interpolation values in y-direction

        # Create a meshgrid of interpolation factors
        U, V = bd.meshgrid(u, v, indexing="ij")  # Shape (sampleX, sampleY)

        # Compute the bilinear interpolation
        grid_points = (
            (1 - U)[..., None] * (1 - V)[..., None]  * self.pointAnchor[0].reshape(1, 1, 3) +
            U[..., None] * (1 - V)[..., None]        * self.pointAnchor[0].reshape(1, 1, 3) +
            (1 - U)[..., None] * V[..., None]        * self.pointAnchor[0].reshape(1, 1, 3) +
            U[..., None] * V[..., None]              * self.pointAnchor[0].reshape(1, 1, 3)
        )  

        # Reshape the matrix into the same dimension as the self.image 
        grid_points = grid_points.reshape(sampleY, sampleX, 3)

        grid_points = bd.concatenate([grid_points, self.image], axis=2)

        self.pointSource = PointsSource(grid_points)



def main():
    
    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    img = Image2D()
    img.imageDimensionOverride = 1280 
    img.LoadFrom8bit(r"resources/ISO12233-4k.png")
    
    RB = img.EmitSamplesToward( targets)


if __name__ == "__main__":
    main() 