
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


def rgbFromRGBA(rgba: bd.ndarray, background=(0, 0, 0)) -> bd.ndarray:
    """
    Convert an RGBA image array (H, W, 4) to an RGB array (H, W, 3),
    compositing transparency over a solid background color.

    :param rgba: bd.ndarray input image, dtype uint8, shape (H, W, 4).
    :param background: Background color (R, G, B), default black.

    :return: RGB image, dtype float32 normalized to [0,1] (ready for plt.imshow()).
    """
    assert rgba.ndim == 3 and rgba.shape[2] == 4, "Input must be (H, W, 4) RGBA array."
    # Normalize to [0,1]
    rgb = rgba[..., :3].astype(bd.float32) / 255.0
    alpha = rgba[..., 3:4].astype(bd.float32) / 255.0
    bg = bd.array(background, dtype=bd.float32) / 255.0

    # Alpha blending: C_out = alpha * C_fg + (1 - alpha) * C_bg
    out = alpha * rgb + (1 - alpha) * bg
    return out  # shape (H, W, 3), float32 in [0,1]


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




