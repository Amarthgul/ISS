

import matplotlib.pyplot as plt


# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
from .Points import PointsSource

from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, RNG
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor
from Util.Misc import Magnitude, ArrayRotate, RectPath
from Raytracing.RayBatch import RayBatch





class Image2D:
    def __init__(self):
        """RGB array directly decoded from the file representing the image"""

        # This class is very much an inherited class from PointSource 
        # But for easier implementation they are still separated. 
        
        self.rgbArray = None 

        """Original image file"""
        self._fileMaster = None 

        """Point source object built from the image"""
        self.pointSource = None

        """Point source object built from parts of the image that's higher than the standard clipping value"""
        self.pointSourceHigh = None

        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 

        """Modifiers that could alter the RayBatch"""
        self.emissionModifier = []

        """Other channels when reading from an EXR file."""
        self.AOVs = None

        """Names of the extra EXR AOV channels, in the same order as self.AOVs."""
        self.AOVNames = []

        """Names of AOV columns packed into pointSource / emitted RayBatch rows."""
        self.pointSourceAOVNames = []

        """An array of same size with the number of point sources. Each entry in this array corresponds to the """
        self.jitterPerPoint = None

        """Same as jitterPerPoint but for the pointSourceHigh"""
        self.jitterPerPointHigh = None


    def AppendAOV(self, name, values):
        """
        Add an AOV to the image.

        :param name: The name of the AOV channel.
        :param values: The values of the AOV channel as an array same size as the image,
            or a scalar value to flood the whole image.

        """
        if self.rgbArray is None:
            raise RuntimeError("Image2D.AppendAOV(): rgbArray is empty. Load an image first.")

        if name is None or str(name) == "":
            raise ValueError("Image2D.AppendAOV(): name must be a non-empty string.")

        image_shape = self.rgbArray.shape[:2]
        pixel_count = image_shape[0] * image_shape[1]

        aov_values = bd.asarray(values, dtype=PRECISION_TYPE)

        if aov_values.ndim == 0:
            aov_values = bd.ones(image_shape, dtype=PRECISION_TYPE) * aov_values
        elif aov_values.ndim == 1:
            if aov_values.shape[0] != pixel_count:
                raise ValueError(
                    "Image2D.AppendAOV(): 1-D values must have one entry per image pixel."
                )
            aov_values = aov_values.reshape(image_shape)
        elif aov_values.ndim == 2:
            if aov_values.shape != image_shape:
                raise ValueError(
                    f"Image2D.AppendAOV(): values shape {aov_values.shape} does not match image shape {image_shape}."
                )
        elif aov_values.ndim == 3 and aov_values.shape[:2] == image_shape and aov_values.shape[2] == 1:
            aov_values = aov_values[:, :, 0]
        else:
            raise ValueError(
                "Image2D.AppendAOV(): values must be scalar, flat per-pixel, HxW, or HxWx1."
            )

        if self.AOVs is None:
            self.AOVs = {}

        if name not in self.AOVNames:
            self.AOVNames.append(name)

        self.AOVs[name] = aov_values.astype(PRECISION_TYPE, copy=False)

        if self.pointSource is not None:
            if hasattr(self, "_GeneratePolarPointSources"):
                self._GeneratePolarPointSources(appendAOV=True)
            elif hasattr(self, "_GeneratePointSources"):
                self._GeneratePointSources()

            if self.pointSourceHigh is not None:
                self.ConstructHighlightPoints()

        return self


    def ReorderAOV(self, nameOrder=None):
        """
        Reorder stored AOV channels by name.

        :param nameOrder: list of AOV names in the desired order. It must contain
            exactly the same names as the current stored AOV channels.
        :return: self
        """
        if nameOrder is None:
            nameOrder = []

        nameOrder = list(nameOrder)

        if self.AOVs is None or len(self.AOVNames) == 0:
            if len(nameOrder) != 0:
                raise ValueError("Image2D.ReorderAOV(): cannot reorder an image with no AOVs.")
            return self

        currentNames = list(self.AOVNames)

        if len(nameOrder) != len(currentNames):
            raise ValueError(
                "Image2D.ReorderAOV(): nameOrder must contain exactly the current AOV names."
            )

        if len(set(nameOrder)) != len(nameOrder):
            raise ValueError("Image2D.ReorderAOV(): nameOrder contains duplicate AOV names.")

        missing = [name for name in currentNames if name not in nameOrder]
        extra = [name for name in nameOrder if name not in currentNames]
        if missing or extra:
            raise ValueError(
                f"Image2D.ReorderAOV(): nameOrder mismatch. Missing={missing}, extra={extra}."
            )

        self.AOVNames = nameOrder
        self.AOVs = {name: self.AOVs[name] for name in self.AOVNames}

        if self.pointSource is not None:
            if hasattr(self, "_GeneratePolarPointSources"):
                self._GeneratePolarPointSources(appendAOV=True)
            elif hasattr(self, "_GeneratePointSources"):
                self._GeneratePointSources()

            if self.pointSourceHigh is not None:
                self.ConstructHighlightPoints()

        return self


    def GetAOVNames(self):
        """
        Return the names of AOV columns packed into emitted rays.

        The order exactly matches the AOV columns generated by
        _GeneratePolarPointSources(appendAOV=True), so names and values can be
        zipped safely by downstream output code.
        """
        return list(self.pointSourceAOVNames)


    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.pixelPitch)


    def GenerateSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=5):
        """
        This generate a series of spots from axis to off axis. 
        The outer-most is defined by x and y field anfle. 
        """
        self.pointSource = PointsSource()
        self.pointSource.GenerateSpots(xAngle, yAngle, dist, sampleField)


    def GetSampleRatios(self):
        
        return self.pointSource.GetSampleRatios()


    def AttachModifier(self, modifier):


        self.emissionModifier.append(modifier)


    def DrawImage(self):
        """
        Draw the points sources in 3D space with corresponding colors.
        """
        DrawPointsPerColor(self.pointSource.Position(), self.pointSource.DisplayColor())


    def Show2D(self, ax=None, show=True, title=None):
        """
        Display the 2D image using matplotlib imshow.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into. If None, a new figure is created.
        show : bool
            Whether to call plt.show() automatically.
        title : str, optional
            Title of the plot.
        """

        if self.rgbArray is None:
            raise RuntimeError("Image2D.Show2D(): rgbArray is empty. Load an image first.")

        # Convert backend array to NumPy for matplotlib if needed
        img = self.rgbArray
        if hasattr(img, "get"):  # CuPy → NumPy
            img = img.get()

        # Handle optional opacity
        if self._opacityArray is not None:
            alpha = self._opacityArray
            if hasattr(alpha, "get"):
                alpha = alpha.get()
            img = bd.concatenate([img, alpha[..., None]], axis=-1)

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(img, origin="upper")
        ax.axis("off")

        if title is not None:
            ax.set_title(title)

        if show:
            plt.show()

        return img


    def ConstructHighlightPoints(self, threshold=1):
        """
        Create a highlight-only point source and its matching jitter array.

        A source is considered a highlight if any of its RGB channels exceeds
        the given threshold. The same boolean mask is applied to both the
        point-source rows and jitterPerPoint so they remain in direct
        correspondence.

        :param threshold: highlight threshold applied to RGB values
        :return: self.pointSourceHigh
        """

        self.pointSourceHigh = None
        self.jitterPerPointHigh = None

        if self.pointSource is None or self.pointSource.value is None:
            return None

        point_values = self.pointSource.value

        # RGB is always stored in columns 3:6 for PointsSource
        rgb = point_values[:, 3:6]

        # Highlight if any channel is above threshold
        highlight_mask = bd.any(rgb > threshold, axis=1)

        highlight_values = point_values[highlight_mask]

        self.pointSourceHigh = PointsSource(highlight_values)
        self.pointSourceHigh.isCartesian = self.pointSource.isCartesian
        self.pointSourceHigh.angleInRad = self.pointSource.angleInRad
        self.pointSourceHigh.emissionPDF = self.pointSource.emissionPDF

        if self.jitterPerPoint is not None:
            self.jitterPerPointHigh = self.jitterPerPoint[highlight_mask]

        return self.pointSourceHigh


    def EmitHighlightSamplesTowards(self, targets, sampleCount=64):

        if self.pointSourceHigh is None:
            self.ConstructHighlightPoints()
            return PointsSource().EmitSamplesToward(targets, 0)

        return self.pointSourceHigh.EmitSamplesToward(targets, sampleCount, self.jitterPerPointHigh)


    def ReceiveAndEmitTowards(self, targets, incidents=None, sampleCount=64, useHighlightSources=False):
        pass




def main():
    from .Image2DFlat import Image2DFlat

    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    img = Image2DFlat()
    img.imageDimensionOverride = 100 
    img.distance = bd.array(100)
    img.LoadFrom8bit(r"resources/Arrow.png")
    
    RB = img.EmitSamplesToward( targets)

    for i in range(len(RB.value)):
        print(RB.value[i])

    SetUnifScale(250)
    AddXYZ()
    DrawRaybatch(RB, lLength=50)
    DrawPoints(targets)
    img.DrawImage()
    plt.show()

    

if __name__ == "__main__":
    main() 

