
import numpy as np 






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



    def SetPoint(self, fX, fY, distance):
        self.fieldX = fX
        self.fieldY = fY
        self.distance = distance 


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
         


def main():
    p = Point()
    p.fieldX = 10
    p.fieldY = 20
    p.distance = 700
    print(p.GetPosition())



if __name__ == "__main__":
    main()
