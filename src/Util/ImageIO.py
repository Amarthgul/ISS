
import os
import OpenEXR

from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Misc import NumpyConversion, RectPath
from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines, RefreshRNG, Axis, UP_DIR, ORIGIN, NEAR_ZERO



def ImageConversion(ary, bitDepth=8, maxModifier=1, normalizer=None, rotate=True):
    """
    Convert the float reprensentation of an image to a uint8 image.
    """
    # print(bd.max(ary))
    maxValue = bd.max(ary) * maxModifier 


    #maxValue = 256
    bits = 2.0**bitDepth-1
    scaleRatio = bits / (maxValue if (normalizer is None) else normalizer + NEAR_ZERO)
    ary = bd.clip(ary*scaleRatio, 0, bits) 

    if(rotate):
        ary = bd.rot90(ary)

    return NumpyConversion(ary).astype(bd.uint8)


def rgbFromRGBA(rgba: bd.ndarray, background=(255, 255, 255)):
    """
    Convert an RGBA image array (H, W, 4) to an RGB array (H, W, 3),
    compositing transparency over a solid background color.

    :param rgba: bd.ndarray input image, dtype uint8, shape (H, W, 4).
    :param background: Background color (R, G, B), default black.

    :return: RGB image, dtype float32 normalized to [0,1] (ready for plt.imshow()).
    """
    import numpy as np

    rgba = np.asarray(rgba)  # ensure NumPy
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    bg = np.array(background, dtype=np.float32).reshape(1, 1, 3) / 255.0

    out = alpha * rgb + (1.0 - alpha) * bg

    return out


def ImageConversionAverage(ary, bitDepth=8, modifier=2, rotate=True):
    """
    Convert the float representation of an image to a uint8 image.
    """
    emanVal = bd.median(ary)

    bits = 2.0**bitDepth-1
    scaleRatio = bits / (emanVal * modifier)
    ary = bd.clip(ary*scaleRatio, 0, bits) 

    if(rotate):
        ary = bd.rot90(ary)

    return NumpyConversion(ary).astype(bd.uint8)


def SaveAsEXR(ary, folder, fileName):
    """
    Save the latent image array as an EXR file. 

    :ary: image array. This will automatically be converted to numpy float32. 
    :param folder: folder location to save the image, note that the folder must exist. 
    :param fileName: the whole file name. 
    """

    RGB = ary.astype(bd.float32)
    if(backend_name == 'cupy'):
        RGB = bd.asnumpy(RGB)

    channels = { "RGB" : RGB }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
            "type" : OpenEXR.scanlineimage }

    folder = RectPath(folder)

    if (folder[-1] != r"/"):
        folder += r"/"

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    nameStr = folder + fileName + ".exr"

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(nameStr)




