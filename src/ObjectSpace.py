
import PIL.Image
import numpy as np 
from src.Util.Misc import * 
import copy


class Image2D:
    def __init__(self, path = None):
        self.imgpath = path 
        self.img = None 
        self._sourceImg = None # A backup of the source image 
        self.bitDepth = 8 # TODO: add bitdepth detection 

        # Angle of view on x, y, and diagonal direction.
        # Sign is based on negative z direction, where clockwise is positive. 
        self.xAoV = np.array([-19.8, 19.8])
        self.yAoV = np.array([-13.5, 13.5])

        self.diagonalAoV = 46.8
        # Dafult values are for a 50mm lens on a 135 imager. 


        # Distance of the input image in millimeter 
        # Default to 1.5m as the normal focus distance of human subjects
        self.distance = 1500 


        # Whether the image will be fit to the AoV. 
        # When flagged, the size will be auto calculated based on AoV and distance 
        self.fitToAoV = True 

        # How to fit the image when aspect ratio is different from AoV  
        self.fitMethod = Fit.FIT
        

        # Physical size of the image in millimeter. 
        self.sizeX = 360 
        self.sizeY = 240 
        # If not fit to angle of view, explicit dimension is needed. 


        # Per mm point sample, when set to <=0, sample will be based on pixel 
        self.samplePerMillimeter = 0

        self.sampleX = 360 # Amount of samples, unitless
        self.sampleY = 240

        # Additional modifier for the amount of sample points.
        # Used to reduce sample proportionally  
        self._sampleModifier = 2

        self.pointData = []


    def Update(self):
        """
        Update the parameters of the image2D object. This method must be manually called before running the system to ensure the parameters are refreshed. 
        """
        if(self.imgpath is not None):
            self.img =  PIL.Image.open(self.imgpath)
            self._sourceImg = copy.deepcopy(self.img)

        _xAoVRad = np.radians(self.xAoV)
        _yAoVRad = np.radians(self.yAoV)
        xTotalAoV = abs(self.xAoV[1] - self.xAoV[0]) # In degrees 
        yTotalAoV = abs(self.yAoV[1] - self.yAoV[0])
        # Update the diagonal 
        self.diagonalAoV = 2 * np.arctan(
            np.sqrt(
                np.tan(xTotalAoV/2)**2 + np.tan(yTotalAoV/2)**2  
                ))
        # Anchor points on x and y direction 
        xAnchors = self.distance * np.tan(_xAoVRad)
        yAnchors = self.distance * np.tan(_yAoVRad)

        #print("Anchor positions ", xAnchors, "  ", yAnchors)

        if(self.fitToAoV):
            self.sizeX = abs(xAnchors[1] - xAnchors[0]) # Full size in mm 
            self.sizeY = abs(yAnchors[1] - yAnchors[0]) 

        
        if(self.samplePerMillimeter == 0):
            # Divides then times 2 to ensure they are even numbers 
            self.sampleX = int(self._sampleModifier * self.sizeX / 2.0) * 2 
            self.sampleY = int(self._sampleModifier * self.sizeY / 2.0) * 2
        else:
            self.sampleX = int(self.sizeX * self.samplePerMillimeter * self._sampleModifier)
            self.sampleY = int(self.sizeY * self.samplePerMillimeter * self._sampleModifier)

        #print("Spatial size  ", self.sizeX, "       ", self.sizeY)
        #print("Sample amount ", self.sampleX, "       ", self.sampleY)
        # Create the sample point grid 
        xLoc = np.linspace(xAnchors[0], xAnchors[1], self.sampleX)
        yLoc = np.linspace(yAnchors[0], yAnchors[1], self.sampleY)
        x, y = np.meshgrid(xLoc, yLoc)
        z = (x * 0) - self.distance
        positions = np.stack((x, y, z), axis=-1)
        #print("x and y: \n", xLoc, "\ny Loc\n", yLoc)

        # Resize the image to the exact sample points 
        self.img = self._sourceImg.resize((self.sampleX, self.sampleY))
        colorArray = self.ImageToRGBArray()

        # An array, each entry is of format (x, y, z, R, G, B)
        self.pointData = np.concatenate((positions, colorArray), axis=-1)


    def SetImage(self, input):
        """
        Override the image content with an external image. 
        """
        self.img = input 
        self._sourceImg = copy.deepcopy(self.img)


    def AreaSpilt(self):
        """
        Spilt this image into 2 parts and return them as 2 Image2D instances with corresponding paramters. 

        :return: 2 Image2D instances of the 2 parts of the original image. 
        """
        # This method is not ideal. While spilting the image could help recursively 
        # propagate the image, it is hard to guarantee that the totoal image dimension
        # remains the same, especially after the second recursive spilt.
        # As a result, it might happen that the final image has a tiny seam due to the
        # 2 image resample not covering the seam in the middle.  

        width, height = self.img.size
        xTotalAoV = self.xAoV[0] + self.xAoV[1]
        yTotalAoV = self.yAoV[0] + self.yAoV[1]

        if(width > height):
            #print("\nWidth spilt")
            p1Box = (0, 0, width // 2, height)    # Left half
            p2Box = (width // 2, 0, width, height)  # Right half
            xAoV1 = np.array([self.xAoV[0], xTotalAoV/2])
            xAoV2 = np.array([xTotalAoV/2, self.xAoV[1]])
            yAoV1 = self.yAoV
            yAoV2 = self.yAoV
        else:
            p1Box = (0, 0, width, height // 2)    # Top half
            p2Box = (0, height // 2, width, height)  # Bottom half
            xAoV1 = self.xAoV
            xAoV2 = self.xAoV
            yAoV1 = np.array([self.yAoV[0], yTotalAoV/2])
            yAoV2 = np.array([yTotalAoV/2, self.yAoV[1]])

        p1 = self.img.crop(p1Box)
        p2 = self.img.crop(p2Box)

        imgP1 = Image2D()
        imgP1.SetImage(p1)
        imgP1.xAoV = xAoV1
        imgP1.yAoV = yAoV1
        imgP1.distance = self.distance
        imgP1.Update()

        imgP2 = Image2D()
        imgP2.SetImage(p2)
        imgP2.xAoV = xAoV2
        imgP2.yAoV = yAoV2
        imgP2.distance = self.distance
        imgP2.Update()

        return imgP1, imgP2 


    def ChannelSpilt(self): 
        """
        Spilt the image into 3 different images representing the RGB channel. 

        :return: 3 Image2D instances made from the original's RGB channel. 
        """
        # Requires the self image to be an RGB image 

        r, g, b = self.img.split()
        
        r = PIL.Image.merge("RGB", (r, PIL.Image.new("L", r.size), PIL.Image.new("L", r.size)))
        g = PIL.Image.merge("RGB", (PIL.Image.new("L", g.size), g, PIL.Image.new("L", g.size)))
        b = PIL.Image.merge("RGB", (PIL.Image.new("L", b.size), PIL.Image.new("L", b.size), b))
        
        imgR = copy.deepcopy(self)
        imgG = copy.deepcopy(self)
        imgB = copy.deepcopy(self)

        for channel, img in zip([r, g, b], [imgR, imgG, imgB]):
            img.SetImage(channel)
            img.Update() 

        return imgR, imgG, imgB
        

    def ImageToRGBArray(self):
        """
        This method converts an image to an array of RGB values. 
        """
        # TODO: add more implementations for images not directly read as RGB
        # For color space conversion and gamma correction, also put them here 
        return np.array(self.img)




class Point:
    """
    A single point source in object space. 
    """
    def __init__(self):

        # The field angles on x and y direction 
        self.fieldX = 0
        self.fieldY = 0

        self.fieldIsDegree = True 

        # Distance is unsigned length in mm counting from the front vertex of the lens
        self.distance = np.inf  

        self.position = np.array([0, 0, 0])
        self.RGB = np.array([0, 0, 0])
        self.bitDepth = 0 # when set to 0, RGB will be in [0, 1] range 


    def _Update(self):
        """
        Calculate the position based on field angles and distance. 
        """
        if (self.fieldIsDegree):
            angleX = np.radians(self.fieldX)
            angleY = np.radians(self.fieldY)
        else:
            angleX = self.fieldX
            angleY = self.fieldY

        oppositeX = self.distance * np.tan(angleX)
        oppositeY = self.distance * np.tan(angleY)

        self.position = np.array([oppositeX, oppositeY, -self.distance])


    def GetPosition(self):
        """
        Check if the position is left at default. If so,
        convert field and distance info into object space XYZ location. 

        :return: XYZ coordiantes of the point source. 
        """
        if(np.dot(self.position, self.position) == 0):
            self._Update() 

        return self.position
         

    def GetColorRGB(self):
        return self.RGB 
    

    def GetBitDepth(self):
        return self.bitDepth



def main():

    isSpotTest = False

    if(isSpotTest):
        p = Point()
        p.fieldX = 10
        p.fieldY = 20
        p.distance = 700
        #print(p.GetPosition())
    else:
        # Henri-Cartier-Bresson.png
        # ISO12233.jpg
        testImgPath = r"resources/Henri-Cartier-Bresson.png"

        testImg = Image2D(testImgPath)
        testImg.xAoV = np.array([-19.8, 0])
        testImg.yAoV = np.array([-13.5, 0])
        testImg.Update()
        testImg.ChannelSpilt()
    



if __name__ == "__main__":
    main()
