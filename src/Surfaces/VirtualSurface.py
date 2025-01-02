



from Util.Backend import backend as bd 
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO

from .Surface import Surface




"""
A virtual surface is something that is non-physical. It is typically used to represent some aspect of the lens, such as the pupil, the principal plane, etc.
"""
class VirtualSurface(Surface):
    def __init__(self):
        super().__init__(0, 0, 0, "AIR")
        self.value = []

        self.cumulativeThickness = None 
        self.frontVertex = None
        self.radiusCenter = None


    def SetData(self, value):
        self.value = value


    def GetPerimeter (self):
        """
        Get the outer perimeter of the virtual surface.
        """

        pass 


