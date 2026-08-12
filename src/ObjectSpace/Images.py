

import matplotlib.pyplot as plt
import PIL.Image


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

        """Horizontal angle of view represented by the image, in degrees."""
        self.horizontalAoV = 40

        """Optional per-pixel opacity in the normalized [0, 1] range."""
        self.alphaArray = None

        """Selected EXR alpha channel name."""
        self.alphaChannelName = None

        """Modifiers that could alter the RayBatch"""
        self.emissionModifier = []

        """Other channels when reading from an EXR file."""
        self.AOVs = None

        """Names of the extra EXR AOV channels, in the same order as self.AOVs."""
        self.AOVNames = []

        """Names of AOV columns packed into pointSource / emitted RayBatch rows."""
        self.pointSourceAOVNames = []

        """Whether non-RGB channels are packed into point sources and emitted rays."""
        self._emitAOVs = False

        """An array of same size with the number of point sources. Each entry in this array corresponds to the """
        self.jitterPerPoint = None

        """Same as jitterPerPoint but for the pointSourceHigh"""
        self.jitterPerPointHigh = None


    @property
    def _opacityArray(self):
        """Compatibility alias for the former flat-image opacity field."""
        return self.alphaArray


    @property
    def emitAOVs(self):
        """Whether this image emits and propagates non-RGB AOV channels."""
        return self._emitAOVs


    @emitAOVs.setter
    def emitAOVs(self, enabled):
        """Enable or disable AOV emission and refresh existing point sources."""
        enabled = bool(enabled)
        if enabled == self._emitAOVs:
            return

        self._emitAOVs = enabled

        if getattr(self, "pointSource", None) is None:
            return

        if hasattr(self, "_GeneratePolarPointSources"):
            self._GeneratePolarPointSources(appendAOV=enabled)
        elif hasattr(self, "_GeneratePointSources"):
            self._GeneratePointSources()

        if getattr(self, "pointSourceHigh", None) is not None:
            self.ConstructHighlightPoints()


    @_opacityArray.setter
    def _opacityArray(self, values):
        self.alphaArray = values


    def LoadFrom8bit(self, imgPath):
        """Load a normalized RGB image and preserve embedded opacity."""
        return self.LoadFrom8bitRGB(imgPath)


    def LoadFrom8bitRGB(self, rgbImgPath):
        """Load RGB and optional opacity from an 8-bit image."""
        return self._Load8bitRGB(rgbImgPath)


    def LoadFrom8BitPNG(self, imgPath):
        """Compatibility entry point for loading an alpha-preserving PNG."""
        return self._Load8bitRGB(imgPath, preserveAlpha=True)


    def LoadFromEXR(self, exrPath,
                    depthChannelNames=("Z", "Z.R", "depth", "Depth.Z", "depth.Z"),
                    alphaChannelNames=("A", "alpha", "Opacity")):
        """Load EXR channels, then delegate class-specific channel handling."""
        exrPath = RectPath(exrPath)
        channels = self._ReadEXR(exrPath, depthChannelNames, alphaChannelNames)
        channels = self._ResizeEXRChannels(channels)

        self._ResetLoadedImageState()
        self._fileMaster = exrPath
        self.rgbArray = channels["rgb"].astype(PRECISION_TYPE)
        self.alphaArray = (
            channels["alpha"].astype(PRECISION_TYPE)
            if channels["alpha"] is not None else None
        )
        self.alphaChannelName = channels["alpha_name"]

        aov_dict = channels["AOVs"]
        self.AOVNames = [name for name in channels["AOVNames"] if name in aov_dict]
        self.AOVs = (
            {name: aov_dict[name].astype(PRECISION_TYPE) for name in self.AOVNames}
            if self.AOVNames else None
        )

        self._LoadEXRFeatures(channels)
        self._EXRLoaded()
        return self


    def EmitTowards(self, targets, sampleCount, flareGlare=False):
        return self.ReceiveAndEmitTowards(
            targets,
            incidents=None,
            sampleCount=sampleCount,
            useHighlightSources=flareGlare,
        )


    def _Load8bitRGB(self, rgbImgPath, preserveAlpha=True, premultiplyAlpha=None):
        rgbImgPath = RectPath(rgbImgPath)
        imageMaster = PIL.Image.open(rgbImgPath)
        return self._Load8bitImage(
            imageMaster,
            preserveAlpha=preserveAlpha,
            premultiplyAlpha=premultiplyAlpha,
        )


    def _Load8bitImage(self, imageMaster, preserveAlpha=True, premultiplyAlpha=None):
        hasAlpha = preserveAlpha and self._ImageHasAlpha(imageMaster)
        imageMaster = imageMaster.convert("RGBA" if hasAlpha else "RGB")
        imageFile = self._ResizePILImage(imageMaster)

        self._ResetLoadedImageState()
        self._fileMaster = imageMaster

        imageArray = bd.array(imageFile).astype(PRECISION_TYPE) / (TWO ** 8 - 1)
        self.rgbArray = imageArray[..., :3]
        self.alphaArray = imageArray[..., 3] if hasAlpha else None

        if premultiplyAlpha is None:
            premultiplyAlpha = self._PremultiplyAlphaOnLoad()
        if premultiplyAlpha and self.alphaArray is not None:
            self.rgbArray = self.rgbArray * self.alphaArray[..., None]

        self._RGBLoaded()
        return self


    def _ResizePILImage(self, image):
        if self.imageDimensionOverride is None:
            return image

        newWidth = int(self.imageDimensionOverride)
        newHeight = int(image.height * (newWidth / image.width))
        return image.resize((newWidth, newHeight))


    def _ImageHasAlpha(self, image):
        return (
            image.mode in ("RGBA", "LA")
            or (image.mode == "P" and "transparency" in image.info)
        )


    def _ResetLoadedImageState(self):
        self.alphaArray = None
        self.alphaChannelName = None
        self.AOVs = None
        self.AOVNames = []
        self.pointSourceAOVNames = []
        self.pointSource = None
        self.pointSourceHigh = None
        self.jitterPerPoint = None
        self.jitterPerPointHigh = None
        self._ResetLoadedImageFeatures()


    def _ReadEXR(self, exrPath, depthChannelNames, alphaChannelNames):
        """Read RGB, depth, alpha, and all remaining EXR channels."""
        import OpenEXR
        import Imath

        exr = OpenEXR.InputFile(exrPath)
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        available = list(header["channels"].keys())

        def read_channel(name):
            if name not in available:
                return None
            arr = bd.frombuffer(exr.channel(name, FLOAT), dtype=bd.float32)
            return arr.reshape(height, width)

        r = read_channel("R")
        g = read_channel("G")
        b = read_channel("B")
        rgb = bd.stack([r, g, b], axis=-1)

        depth = None
        depthName = None
        for name in depthChannelNames:
            depth = read_channel(name)
            if depth is not None:
                depthName = name
                break

        alpha = None
        alphaName = None
        for name in alphaChannelNames:
            alpha = read_channel(name)
            if alpha is not None:
                alphaName = name
                break

        usedNames = {"R", "G", "B"}
        if depthName is not None:
            usedNames.add(depthName)
        if alphaName is not None:
            usedNames.add(alphaName)

        aovNames = [name for name in available if name not in usedNames]
        aovs = {name: read_channel(name) for name in aovNames}

        return {
            "rgb": rgb,
            "r": r,
            "g": g,
            "b": b,
            "alpha": alpha,
            "depth": depth,
            "channels": available,
            "AOVs": aovs,
            "AOVNames": aovNames,
            "depth_name": depthName,
            "alpha_name": alphaName,
        }


    def _ResizeEXRChannels(self, channels):
        if self.imageDimensionOverride is None:
            return channels

        height, width, _ = channels["rgb"].shape
        newWidth = int(self.imageDimensionOverride)
        newHeight = int(height * (newWidth / width))
        yIndex = bd.linspace(0, height - 1, newHeight).astype(bd.int64)
        xIndex = bd.linspace(0, width - 1, newWidth).astype(bd.int64)
        index = bd.ix_(yIndex, xIndex)

        channels["rgb"] = channels["rgb"][index]
        if channels["depth"] is not None:
            channels["depth"] = channels["depth"][index]
        if channels["alpha"] is not None:
            channels["alpha"] = channels["alpha"][index]
        channels["AOVs"] = {
            name: channels["AOVs"][name][index]
            for name in channels["AOVNames"]
        }
        return channels


    # Loading feature hooks. Subclasses implement only the stages they own.
    def _ResetLoadedImageFeatures(self):
        pass


    def _PremultiplyAlphaOnLoad(self):
        return False


    def _RGBLoaded(self):
        pass


    def _LoadEXRFeatures(self, channels):
        pass


    def _EXRLoaded(self):
        pass


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
                self._GeneratePolarPointSources(appendAOV=self.emitAOVs)
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
                self._GeneratePolarPointSources(appendAOV=self.emitAOVs)
            elif hasattr(self, "_GeneratePointSources"):
                self._GeneratePointSources()

            if self.pointSourceHigh is not None:
                self.ConstructHighlightPoints()

        return self


    def GetAOVNames(self):
        """
        Return the names of AOV columns packed into emitted rays.

        The order exactly matches the AOV columns generated by
        the active ``emitAOVs`` setting, so names and values can be
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
        if self.alphaArray is not None:
            alpha = self.alphaArray
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

