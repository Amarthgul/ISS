
import PIL.Image
import numpy as np 
import PIL


class Image2D:
    def __init__(self, path):
        self.imgpath = path 

        self.distance = np.inf 

        # Size of the image in millimeter. 
        self.sizeX = 360 
        self.sizeY = 240 

        # Per mm point sample, when set to <=0, sample will be based on pixel 
        self.samplePerMillimeter = 0

        self.points = [] # List of objects of class Point 

        self._Start() 


    def _Start(self):
        img =  PIL.Image.open(self.imgpath)
        imageDimX = img.size  
        print(imageDimX)


    def GetPoints(self):
        """
        Get a list of all the points in this image. 
        
        :return: an array of objects in the type of Point class. 
        """
        pass 




class Point:
    """
    A single point source in object space. 
    """
    def __init__(self):
        self.fieldX = 0
        self.fieldY = 0
        self.fieldIsDegree = True 

        # Distance is unsigned length in mm counting from the front vertex of the lens
        self.distance = np.inf  

        self.RGB = np.array([0, 0, 0])
        self.bitDepth = 0 # when set to 0, RGB will be in [0, 1] range 


    def GetPosition(self):
        """
        Convert field and distance info into object space XYZ location. 

        :return: XYZ coordiantes of the point source. 
        """
        if (self.fieldIsDegree):
            angleX = np.radians(self.fieldX)
            angleY = np.radians(self.fieldY)
        else:
            angleX = self.fieldX
            angleY = self.fieldY

        oppositeX = self.distance * np.tan(angleX)
        oppositeY = self.distance * np.tan(angleY)

        return np.array([oppositeX, oppositeY, -self.distance])
         

    def GetColorRGB(self):
        return self.RGB 
    

    def GetBitDepth(self):
        return self.bitDepth


def main():

    testImgPath = r"Henri-Cartier-Bresson.png"

    testImg = Image2D(testImgPath)

    p = Point()
    p.fieldX = 10
    p.fieldY = 20
    p.distance = 700
    # print(p.GetPosition())



if __name__ == "__main__":
    main()
