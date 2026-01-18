

"""
This class is the abridged equivalence of the Zemax Analysis tab.
"""

from Lens import Lens


class Analysis():
    def __init__(self, lens, imager):
        self.lens = lens
        self.imager = imager



    def Distortion(self, objectT, focusT=None, maxFieldAngleX=0, maxFieldAngleY=0,  samplePoints=10):
        """
        Calculate the geometric distortion of the system.

        :param objectT: object distance.
        :param focusT: focus distance.
        :param maxFieldAngleX: maximum field angle along X direction.
        :param maxFieldAngleY: maximum field angle along Y direction.
        :param samplePoints: number of spots used to sample.
        """




        pass