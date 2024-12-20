
# For unused functions and tests 

import numpy as np 

from Globals import * 


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
    R, G, B = 0.0, 0.0, 0.0

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
        G = 1 - ratio

    elif (LambdaLines[primaries["B"]] <= wavelength < LambdaLines[primaries["G"]]):
        # Between Blue and Green
        ratio = (wavelength - LambdaLines[primaries["B"]]) / (LambdaLines[primaries["G"]]- LambdaLines[primaries["B"]])
        G = ratio
        B = 1 - ratio

    elif (LambdaLines[UVIRcut[0]] <= wavelength < LambdaLines[primaries["B"]]):
        # Between Blue primary and UV cutoff
        B = 1.0
        G = (wavelength - LambdaLines[UVIRcut[0]]) / (LambdaLines[primaries["B"]] - LambdaLines[UVIRcut[0]])

    # Scale to bit depth and return
    
    if (useBits):
        R = int(np.clip(R * bits, 0, bits))
        G = int(np.clip(G * bits, 0, bits))
        B = int(np.clip(B * bits, 0, bits))

    return np.array([R, G, B])


#  ===========================================================================
"""
==============================================================================
"""

def main():
    RGB = WavelengthToRGB(643.85) 
    #wavelngth = RGBToWavelength(RGB)
    RGB1 = WavelengthToRGB(589.3) 

    print("RGB result: \t", RGB)
    #print("wavelength: \t", wavelngth)
    print("RGB result: \t", RGB1)

#Rotation(0.61, np.array([0, 1, 0]), np.array([1, 1, 0]))
if __name__ == "__main__":
    main() 