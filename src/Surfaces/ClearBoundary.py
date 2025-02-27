
from Util.Backend import backend as bd 
from Util.Backend import constant


class ClearBoundary():

    def __init__(self, d, t):
        self.clearSemiDiameter = constant(d)
        self.cumulativeThickness = constant(t)


    def DrawSurface(self):
        pass 

    