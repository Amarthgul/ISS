

import numpy as np 
import copy 

from Util import *

"""

This file is for things that are questionable and less useful, but not entirely useless. To ditech them entirely is a bit wasteful, and to avoid one day they become needed again, they are placed here.  

"""




def _imageRaybatch(self, image, bitDepth=8, samplePoints = PER_POINT_MAX_SAMPLE):
    """
    This method was used for generating initial raybatch when transitioning to image propagation.
    However, it becomes rather questionable if such method is actually needed. An even distribution can also be obtained by reducing the radiant of oblique rays, rather than actually trying to find a 3D distribution on the direction of entrance pupil. 
    """
    lumi = LumiPeakArray(image.ImageToRGBArray())
    print(lumi.shape)

    wavelengths, radiants = RGBToWavelength(image.ImageToRGBArray(), self.primaries, self.secondaries, self.UVIRcut)

    firstSurface = self.lens.surfaces[0] 
    r = firstSurface.radius 
    sd = firstSurface.clearSemiDiameter 

    posP = image.pointData[:, :, :3] # First 3 entries are the positions 
    
    P_xy_projection = copy.deepcopy(posP)
    P_xy_projection[:, :, 2] = 0  # Projection on xy plane 

    offset = np.array([0, 0, abs(r) - np.sqrt(r**2 - sd**2)]) * np.sign(r)
    
    norms = np.linalg.norm(P_xy_projection, axis=2, keepdims=True)
    is_zero = np.isclose(norms, 0)
    normalized_projection = np.where(is_zero, 0, P_xy_projection / norms)

    # Grid of point A and point C for every sample points 
    posA = np.where(is_zero, 
                    np.array([sd, 0, 0]) + offset, 
                    sd * normalized_projection + offset)
    posC = np.where(is_zero, 
                    np.array([sd, 0, 0]) + offset, 
                    sd * -normalized_projection + offset)
    
    posB = self._findBArray(posA, posC, posP)

    sampleAmountLumi = np.array(lumi * samplePoints).astype(int)

    points = np.transpose(self._ellipsePeripheral(posA, posB, posC, posP, sd, r, sampleAmountLumi)) # Sample points in the ellipse area 
    # vecs = ArrayNormalized(points - posP)

    print("somrhitng ")
