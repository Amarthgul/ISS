
"""
This is for all the color and wavelength calculations and conversions. 


"""


import math


from Util.Misc import NumpyConversion
from Util.Backend import backend as bd 
from Util.Backend import backend_name
from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines, RefreshRNG, Axis, UP_DIR, ORIGIN, NEAR_ZERO




def LumiPeak(RGB, bitDepth = 8):
    """
    Naive solution for calculating the luminance based on the RGB channel of a pixel/point. 

    :param RGB: RGB value of a pixel as [R, G, B]. 
    :param bitDepth: bitDepth if the RGB array is not in the [0, 1] range. 
    """
    if(bd.sum(RGB) > 3):
        RGB = RGB / (2**bitDepth)

    lumi = 0.2126*RGB[0] + 0.7152*RGB[1] + 0.0722 *RGB[2]
    return lumi


def Lumi(RGB):
    """
    Accquiring the lumosity of the inputs by weighted average of RGB
    """
    return 0.2126*RGB[:, 0] + 0.7152*RGB[:, 1] + 0.0722 *RGB[:, 2]


def LumiPeakArray(RGB, bitDepth = 8):
    """
    Naive solution for calculating the luminance based on the RGB channel of an image/array.  

    :param RGB: RGB array in the shape of (m, n, 3). 
    :param bitDepth: bitDepth if the RGB array is not in the [0, 1] range.
    """
    if(RGB.max() > 1):
        RGB = RGB / (2**bitDepth)

    return 0.2126*RGB[:, :, 0] + 0.7152*RGB[:, :, 1] + 0.0722 *RGB[:, :, 2] 


def RGBToWavelength(RGB, 
                    primaries = {"R": "C'", "G": "e", "B":"g"}, 
                    secondaries = ["F", "D"], 
                    UVIRcut = ["i", "A'"],
                    bitDepth=8):
    """
    Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
    
    :param RGB: RGB values
    :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"})
    :param secondaries: A dictionary mapping secondary colors to wavelength lines (optional)
    :param UVIRcut: Cut wavelength for ultraviolet and infrared, the first term is UV and the second is IR. 
    :return: A NumPy array of wavelengths corresponding to the ibdut RGB array
    """

    # Normalize RGB values to the range [0, 1]
    bits = 2.0 ** bitDepth - 1

    wavelengths = bd.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])

    radiants = bd.array(RGB)
    #print("INSIDE RAD", radiants)

    if (len(secondaries) > 0):
        for secondary in secondaries:
            if(type(secondary) is str):
                currentWavelength = LambdaLines[secondary]
            else:
                currentRadiant = secondary # When passed as numbers 
            currentRadiant = 0
            wavelengths = bd.append(wavelengths, currentWavelength)

            # Between IR limit and Red line 
            if(currentWavelength < LambdaLines[UVIRcut[1]] and currentWavelength > LambdaLines[primaries["R"]]):
                # Using red radiant and reduce the intensity depending on how far it is away from the red line 
                currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[primaries["R"]]) / (LambdaLines[UVIRcut[1]] - LambdaLines[primaries["R"]]) )

            # Between Red line and Green line 
            elif(currentWavelength < LambdaLines[primaries["R"]] and currentWavelength > LambdaLines[primaries["G"]]):
                # Find the ratio between red and green 
                ratio = (currentWavelength - LambdaLines[primaries["G"]]) / (LambdaLines[primaries["R"]] - LambdaLines[primaries["G"]])

                currentRadiant = radiants[0] * ratio + radiants[1] * (1 - ratio)

            # Between Green line and Blue line 
            elif(currentWavelength < LambdaLines[primaries["G"]] and currentWavelength > LambdaLines[primaries["B"]]):
                # Find the ratio between green and blue 
                ratio = (currentWavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]] - LambdaLines[primaries["B"]])

                currentRadiant = radiants[1] * ratio + radiants[2] * (1 - ratio)

            # Between Blue line and UV limit 
            elif(currentWavelength < LambdaLines[primaries["B"]] and currentWavelength > LambdaLines[UVIRcut[0]]):
                currentRadiant = radiants[0] * ( (currentWavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]]) )

            radiants = bd.append(radiants, currentRadiant)

    return (wavelengths, radiants)


def RGBToWavelengthArray(RGB, 
                primaries = {"R": "C'", "G": "e", "B":"g"}, 
                secondaries = ["F", "D"], 
                UVIRcut = ["i", "A'"],
                bitDepth=8):
    """
    Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
    
    :param RGB: a 3D array in shape (m, n, 3) representing the RGB of an image. 
    :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"}).
    :param secondaries: A dictionary mapping secondary colors to wavelength lines (optional)
    :param UVIRcut: Cut wavelength for ultraviolet and infrared, the first term is UV and the second is IR. 

    :return: A NumPy array of wavelengths corresponding to the input RGB array. 
    """

    if(RGB.max() > 1):
        RGB = RGB / (2**bitDepth)

    width = RGB.shape[0]
    height = RGB.shape[1]

    wavelengths = bd.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])
    wavelengths = bd.tile(wavelengths, (width, height, 1))

    radiants = bd.array(RGB)

    # TODO: Add secondary support? 

    return (wavelengths, radiants)


def RGBToWavelengthSameD(RGB, 
                primaries = {"R": "C'", "G": "e", "B":"g"}, 
                bitDepth=8):
    """
    Convert an RGB values to corresponding wavelengths and intensity/radiant flux.
    
    :param RGB: an array in shape (n, 3) representing many RGB values. 
    :param primaries: A dictionary mapping RGB to primary wavelength lines (default: {"R": "C'", "G": "e", "B": "g"}).
    :param bitDepth: if input is RGB value from an image, the value is likely higher than 1. This parameter will be used to convert the values to [0, 1]. 

    :return: An array of wavelengths corresponding to the input RGB array. 
    """

    if(RGB.max() > 1):
        RGB = RGB / (2**bitDepth)

    wavelengths = bd.array([
        LambdaLines[primaries["R"]], 
        LambdaLines[primaries["G"]], 
        LambdaLines[primaries["B"]]
    ])
    wavelengths = bd.tile(wavelengths, (RGB.shape[0], 1))

    radiants = bd.array(RGB)

    return (wavelengths, radiants)


def RGBToWavelengthSpotSim(RGB, 
                primaries = {"R": "C'", "G": "e", "B":"g"}, 
                bitDepth=8, addCount=2):
    
    return AddSecondary(
        *RGBToWavelengthSameD(RGB, primaries, bitDepth),
         addCount=addCount)


def WavelengthToRGB(wavelength, 
                    primaries={"R": "C'", "G": "e", "B": "g"},
                    UVIRcut=["i", "A'"],
                    useBits=False, bitDepth=8):
    """
    Convert a wavelength to RGB values.
    
    :param wavelength: Wavelength to convert (in nm).
    :param primaries: A dictionary mapping RGB to primary wavelength lines.
    :param UVIRcut: List with UV and IR limits for cutoff.
    :param bitDepth: Bit depth for RGB values.
    :return: RGB values as integers in the range [0, 255].
    """

    # TODO: Note that this function works in conjunction with RGBToWavelength, \
    # thus should be edited together with RGBToWavelength to ensure the results. 
    # For this reason, there is no secondaries here since they are just linear interpolations and can be just ignored  

    # Default RGB values (intensity normalized to 0-1)
    R, G, B = ZERO, ZERO, ZERO

    # Calculate bit depth scaling factor
    bits = 2 ** bitDepth - 1

    # Check where the wavelength falls and interpolate
    if (LambdaLines[primaries["R"]] <= wavelength <= LambdaLines[UVIRcut[1]]):
        # Between Red primary and IR cutoff
        R  = (LambdaLines[UVIRcut[1]] - wavelength) / (LambdaLines[UVIRcut[1]] - LambdaLines[primaries["R"]])

    elif (LambdaLines[primaries["G"]] <= wavelength < LambdaLines[primaries["R"]]):
        # Between Green and Red
        ratio = (wavelength - LambdaLines[primaries["G"]]) / (LambdaLines[primaries["R"]] - LambdaLines[primaries["G"]])
        R = ratio
        G = ONE - ratio

    elif (LambdaLines[primaries["B"]] <= wavelength < LambdaLines[primaries["G"]]):
        # Between Blue and Green
        ratio = (wavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]]- LambdaLines[primaries["B"]])
        G = ratio
        B = ONE - ratio

    elif (LambdaLines[UVIRcut[0]] <= wavelength < LambdaLines[primaries["B"]]):
        # Between Blue primary and UV cutoff
        B = ONE
        G = (wavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]])


    R, G, B = bd.array(R), bd.array(G), bd.array(B)

    if (useBits):
        return bd.clip(bd.array([R, G, B]) * bits, ZERO, bits)
    else:
        return bd.array([R, G, B])

        
def ColorTuplePLT(arrayRGB):
    if(backend_name == 'cupy'):
        arrayRGB = bd.asnumpy(arrayRGB)

    return tuple(arrayRGB)


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


def AddSecondary(wavelength, radiant, addCount=2):
    """
    This method is specifically designed to interpolate and add more secondary spectrums into the wavelength-radiant pair so that spot simulation would become more accurate, especially in the higher angle fields. It does not offer too much improvement for imaging applicaiton. 
    """
    
    if (addCount == 0):
        return wavelength, radiant

    # 3 default wavelenths for RGB plus the newer secondary wavelengths 
    targetColumns = 3 + addCount * 2

    def interpolate_row(row, target_columns):
        x_original = bd.linspace(0, 1, len(row))  # Original x-coordinates
        x_new = bd.linspace(0, 1, target_columns)  # New x-coordinates for interpolation
        return bd.interp(x_new, x_original, row)
    
    wlAppended =bd.array([interpolate_row(row, targetColumns) for row in wavelength])
    raAppended = bd.array([interpolate_row(row, targetColumns) for row in radiant])

    deno = addCount+1
    normalizer = bd.array(1 / (deno)) * bd.ones(targetColumns)
    tailModifier = bd.sum(bd.arange(addCount+2)) / deno
    normalizer[0] *= tailModifier
    normalizer[-1] *= tailModifier
    raAppended = bd.array([normalizer * row for row in raAppended])

    # divs = addCount + 1
    # totalAdd = bd.sum(bd.arange(addCount+1)) / divs
    # greenTotal = totalAdd * 2 + 1

    # normalizer = bd.ones(targetColumns)
    # normalizer[0] += (greenTotal - totalAdd - 1)
    # normalizer[-1] += (greenTotal - totalAdd - 1)
    # normalizer /= normalizer[0]
    # raAppended = bd.array([normalizer * row for row in raAppended])


    return wlAppended, raAppended







def main():
    points = bd.array([[1, 0, 0], [1, 2, 3]])
    colors = bd.array([[1, 1, 1], [1, 0.5, 0.25]])

    print(RGBToWavelengthSpotSim(colors, addCount=2))
    print(WavelengthToRGB(490.955))




if __name__ == "__main__":
    main()