
import PIL.Image
import numpy as np 
from Util import * 


class Image2D:
    def __init__(self, path = None):
        self.imgpath = path 
        self.img = None 
        self.bitDepth = 8 # TODO: add bitdepth detection 

        # Angle of view on x, y, and diagonal direction.
        # Dafult values are for a 50mm lens on a 135 imager. 
        self.xAoV = 39.6
        self.yAoV = 27.0
        self.diagonalAoV = 46.8

        # Distance of the input image in millimeter 
        # Default to 1.5m as the normal focus distance of human subjects
        self.distance = 1500 


        # Whether the image will be fit to the AoV. 
        # When flagged, the size will be auto calculated based on AoV and distance 
        self.fitToAoV = True 

        # How to fit the image when aspect ratio is different from AoV  
        self.fitMethod = Fit.FIT
        
        # If not fit to angle of view, explicit dimension is needed 
        # Size of the image in millimeter. 
        self.sizeX = 360 
        self.sizeY = 240 

        # Per mm point sample, when set to <=0, sample will be based on pixel 
        self.samplePerMillimeter = 0

        self.sampleX = 360 
        self.sampleY = 240

        # Additional modifier for the amount of sample points.
        # Used to reduce sample proportionally  
        self._sampleModifier = 0.5

        self.pointData = []

        self.Update() 


    def Update(self):
        self.img =  PIL.Image.open(self.imgpath)

        if(self.fitToAoV):
            angleX = np.radians(self.xAoV/2.0)  # With half angle division 
            angleY = np.radians(self.yAoV/2.0)
            self.sizeX = self.distance * math.tan(angleX) * 2.0 # Full size 
            self.sizeY = self.distance * math.tan(angleY) * 2.0


        if(self.samplePerMillimeter == 0):
            # Divides then times 2 to ensure they are even numbers 
            self.sampleX = int(self._sampleModifier * self.sizeX / 2.0) * 2 
            self.sampleY = int(self._sampleModifier * self.sizeY / 2.0) * 2
        else:
            self.sampleX = int(self.sizeX * self.samplePerMillimeter * self._sampleModifier)
            self.sampleY = int(self.sizeY * self.samplePerMillimeter * self._sampleModifier)

        #print(self.sizeX, "       ", self.sizeY)
        #print(self.sampleX, "       ", self.sampleY)

        # Create the sample point grid 
        xLoc = np.linspace(-self.sizeX/2, self.sizeX/2, self.sampleX)
        yLoc = np.linspace(-self.sizeY/2, self.sizeY/2, self.sampleY)
        x, y = np.meshgrid(xLoc, yLoc)
        z = (x * 0) - self.distance
        positions = np.stack((x, y, z), axis=-1)

        # Resize the image to the exact sample points 
        imgResize = self.img.resize((self.sampleX, self.sampleY))
        colorArray = self._ImageToRGB(imgResize)

        # An array, each entry is of format (x, y, z, R, G, B)
        self.pointData = np.concatenate((positions, colorArray), axis=-1)


    def SetImage(self, input):
        self.img = input 


    def _ImageToRGB(self, image):
        """
        This method converts an image to an array of RGB values. 
        """
        # TODO: add more implementations for images not directly read as RGB
        # For color space conversion and gamma correction, also put them here 
        return np.array(image)




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

    testImgPath = r"resources/Henri-Cartier-Bresson.png"

    testImg = Image2D(testImgPath)

    p = Point()
    p.fieldX = 10
    p.fieldY = 20
    p.distance = 700
    #print(p.GetPosition())



if __name__ == "__main__":
    main()
