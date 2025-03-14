

import OpenEXR


from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Misc import NumpyConversion
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


def ImageConversionAverage(ary, bitDepth=8, modifier=2, rotate=True):
    """
    Convert the float reprensentation of an image to a uint8 image.
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
    
    if(folder[-1] != r"/"):
        folder += r"/"

    nameStr = folder + fileName + ".exr"

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(nameStr)


