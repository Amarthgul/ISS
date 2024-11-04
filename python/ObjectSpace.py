
import numpy as np 




class Point:
    def __init__(self):
        self.fieldX = 0
        self.fieldY = 0
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
        pass 


