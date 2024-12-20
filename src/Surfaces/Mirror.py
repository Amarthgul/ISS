

from .Surface import * 

class Mirror(Surface): 
    """
    Mirror surface. Reflects rays instead of refracting them. 
    """
    def __init__(self):
        # if this value is set to None Zero 
        self.nullRadius = None 

        # When flagged, thickness will be treated as the absolute value,
        # i.e., replacing the cumulativeThickness attribute 
        self.isAbsoluteT = False 

