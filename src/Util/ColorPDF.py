
"""
Probability Density Function for color-wavelength conversion.

"""



from Util.Globals import RNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines


class ColorPDF:

    def __init__(self):
        # These Fraunhofer lines are also the mu of the Gaussian distribution
        self.lineR = "C'"
        self.lineG = "e"
        self.lineB = "g"

        # Sigma parameter for the three Gaussian respectively
        self.sigmaR = 30
        self.sigmaG = 20
        self.sigmaB = 20

        # Alpha set is for skewed Gaussian, when set to 0 they're just standard Gaussian
        self.alphaR = 0
        self.alphaG = 0
        self.alphaB = 0


    def ColorToWavelength(self, colors, perChannelSample=2048):
        """
        Convert many colors into a batch of wavelengths.

        :param colors: array of RGB colors in shape (m, 3), values in range [0, inf]
        :param perChannelSample: number of samples per color channel.
        """

        #


        # Return an array of
        pass