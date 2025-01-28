



from Util.Backend import backend as bd
from Util.Globals import ONE, INIT_PHASE_DIFF
from Raytracing.RayBatch import RayBatch



class Image2D:
    def __init__(self, imageRGB=None):
        self.imageRGB = imageRGB

        """Float array of the RGB data"""
        self.imageSource = None



    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

