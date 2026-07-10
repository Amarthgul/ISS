

import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR

from src.Raytracing.RayBatch import RayBatch
from .ImageVariDepth import Image2DVariDepth
from .Images import Image2D
from .Image2DFlat import Image2DFlat
from Util.Backend import backend as bd
from Util.Globals import FAR_DISTANCE

import warnings

class ImageStack:
    def __init__(self):

        self.images = {}

        self.layers = 0


    def AddImage(self, image:Image2D, nameTag:str="Img"):
        """Images will be added like a stack, the first added will be at the bottom-most, i.e., furthest from the camera system in the image space.
        :param image: An image class object based on Image2D or its inherited classes.
        :param nameTag: Name tag for the image for easier recognition.
        """

        print("Max min of ", nameTag, ": ",  bd.max(image.rgbArray), ", ", bd.min(image.rgbArray))

        if nameTag == "Img":
            nameTag = "Img"+str(self.layers)

        elif nameTag in self.images:
            warnings.warn("An image with the same tag name already exists, this one will be overwritten.")

        self.images[nameTag] = image


    def AlphaImage(self, image, opaChannelName="X"):
        """
        Set the instance as a single image simulation stack.
        A flat background image will be created, and a new channel named
        opaChannelName will be added to both images' AOV. The input image will
        have value 0 and the flat background will have value 1.
        """

        if image is None:
            raise ValueError("ImageStack.AlphaImage(): image cannot be None.")
        if getattr(image, "rgbArray", None) is None:
            raise RuntimeError("ImageStack.AlphaImage(): image.rgbArray is empty. Load an image first.")
        if opaChannelName is None or str(opaChannelName) == "":
            raise ValueError("ImageStack.AlphaImage(): opaChannelName must be a non-empty string.")

        sampleY, sampleX = image.rgbArray.shape[:2]

        background = Image2DFlat()
        background.distance = FAR_DISTANCE
        background.horizontalAoV = getattr(image, "horizontalAoV", background.horizontalAoV)
        background.imageDimensionOverride = sampleX

        # White carries the opacity AOV through the normal image-emission sampler.
        background._fileMaster = PIL.Image.new("RGB", (int(sampleX), int(sampleY)), (255, 255, 255))
        background._Update()

        background.AppendAOV(opaChannelName, 1)
        #image.rgbArray = bd.zeros_like(image.rgbArray, dtype=image.rgbArray.dtype)
        image.rgbArray = bd.ones_like(image.rgbArray, dtype=image.rgbArray.dtype)
        image.AppendAOV(opaChannelName, 0)

        self.images = {}
        self.layers = 0
        self.AddImage(background, "AlphaBackground")
        self.AddImage(image, "Image")
        self.UnifyAOV(fillerValues={"default": 0, opaChannelName: 0})

        return background


    def UnifyAOV(self, fillerValues = 0):
        """
        Ensure all images in the stack emit AOV columns in the same order.

        Missing AOV channels are added to an image with ``fillerValues``. The
        final order is the stack-wide union of emitted AOV names. Implicit
        vari-depth AOVs such as alpha/depth remain at the front, and explicit
        image AOVs are reordered through Image2D.ReorderAOV.

        :param fillerValues: scalar filler, dict keyed by AOV name, or list/tuple
            aligned to the unified AOV order.
        :return: unified AOV name list.
        """

        unifiedNames = self.GetAOVNames()
        if len(unifiedNames) == 0:
            return unifiedNames

        for _tag, image in self.images.items():
            currentNames = self._ImageAOVNames(image)

            for name in unifiedNames:
                if name not in currentNames:
                    image.AppendAOV(name, self._FillerValue(name, unifiedNames, fillerValues))
                    currentNames = self._ImageAOVNames(image)

            storedOrder = [name for name in unifiedNames if name in getattr(image, "AOVNames", [])]
            image.ReorderAOV(storedOrder)

            emittedNames = self._ImageAOVNames(image)
            if emittedNames != unifiedNames:
                raise RuntimeError(
                    f"ImageStack.UnifyAOV(): image '{_tag}' emits AOVs in order {emittedNames}, "
                    f"not unified order {unifiedNames}."
                )

        return unifiedNames


    def GetAOVNames(self):
        """
        Return the stack-wide union of AOV names.

        Implicit image AOVs such as vari-depth alpha/depth are placed first
        because those channels are emitted before explicit appended AOVs by the
        image classes.
        """

        implicitNames = []
        explicitNames = []

        for _tag, image in self.images.items():
            implicit = self._ImageImplicitAOVNames(image)

            for name in implicit:
                if name not in implicitNames:
                    implicitNames.append(name)

            for name in self._ImageAOVNames(image):
                if name in implicit:
                    continue
                if name not in explicitNames:
                    explicitNames.append(name)

        return implicitNames + [name for name in explicitNames if name not in implicitNames]


    def PrintLayerTags(self):
        for index, key in enumerate(self.images.keys()):
            print(index, key)


    def EmitTowards(self, targets, sampleCount, flareGlare=False):

        wholeRB = RayBatch()

        for key, currentImage in self.images.items():
            wholeRB = currentImage.ReceiveAndEmitTowards(
                    targets,
                    wholeRB,
                    sampleCount,
                    useHighlightSources=flareGlare)

        return wholeRB


    # ==================================================================
    """ ============================================================ """
    # ==================================================================


    def _ImageImplicitAOVNames(self, image):
        emittedNames = []
        if hasattr(image, "GetAOVNames"):
            emittedNames = list(image.GetAOVNames())

        storedNames = list(getattr(image, "AOVNames", []))
        implicitNames = [name for name in emittedNames if name not in storedNames]

        if len(implicitNames) == 0:
            if getattr(image, "alphaArray", None) is not None:
                implicitNames.append(getattr(image, "alphaChannelName", None) or "A")

            if getattr(image, "zArray", None) is not None:
                implicitNames.append(getattr(image, "depthChannelName", None) or "Z")

        return implicitNames


    def _ImageAOVNames(self, image):
        """
        Return the AOV names this image would emit, including implicit depth
        image channels when point sources have not been regenerated yet.
        """

        names = []
        if hasattr(image, "GetAOVNames"):
            names = list(image.GetAOVNames())

        if len(names) == 0:
            if getattr(image, "alphaArray", None) is not None:
                names.append(getattr(image, "alphaChannelName", None) or "A")

            if getattr(image, "zArray", None) is not None:
                names.append(getattr(image, "depthChannelName", None) or "Z")

            for name in getattr(image, "AOVNames", []):
                if name not in names:
                    names.append(name)

        return names


    def _FillerValue(self, name, unifiedNames, fillerValues):
        if isinstance(fillerValues, dict):
            if name in fillerValues:
                return fillerValues[name]
            if "default" in fillerValues:
                return fillerValues["default"]
            return 0

        if isinstance(fillerValues, (list, tuple)):
            return fillerValues[unifiedNames.index(name)]

        return fillerValues




def ExampleStack3D(horizontalAoV=40):

    FG = Image2DVariDepth()
    FG.horizontalAoV = horizontalAoV
    FG.zFarLimit = 1e3
    FG.LoadFromEXR(r"resources/DepthSceneFG.exr")

    # FG.DrawMask()

    MG = Image2DVariDepth()
    MG.horizontalAoV = horizontalAoV
    MG.LoadFromEXR(r"resources/DepthSceneMG.exr")
    print("MG Stats ======================")
    print(MG.Stats())

    MG2 = Image2DVariDepth()
    MG2.horizontalAoV = horizontalAoV
    MG2.LoadFromEXR(r"resources/DepthSceneMG2.exr")
    print("MG2 Stats ======================")
    print(MG2.Stats())

    BG = Image2DVariDepth()
    BG.horizontalAoV = horizontalAoV
    BG.LoadFromEXR(r"resources/DepthSceneBG.exr")
    BG.FloodDepth(2000000.0)
    print("BG Stats ======================")
    print(BG.Stats())


    # FG.DrawMask()

    exampleStack = ImageStack()
    exampleStack.AddImage(BG, "BG")
    exampleStack.AddImage(MG2, "MG2")
    exampleStack.AddImage(MG, "MG")
    exampleStack.AddImage(FG, "FG")

    exampleStack.PrintLayerTags()

    return exampleStack


def ExampleStack2D():
    from .ImageExt import Image2DVariHighlightExtension, Image2DFlatHighlightExtension

    FG = Image2DFlatHighlightExtension()
    FG.zDistance = 900
    FG.LoadFrom8bitRGB(r"resources/2DFrameExample_FG.png")

    MG = Image2DVariHighlightExtension()
    MG.zDepthMappingRange = [1000, 1500]
    MG.LoadFrom8bitRGB(r"resources/2DFrameExample_MG.png")
    MG.LoadFrom8bitZ(r"resources/2DFrameExample_MGZ.png")
    MG.ReconstructHighlight()
    MG.UpdatePointSources()

    BG1 = Image2DFlatHighlightExtension()
    BG1.zDistance = 35000
    BG1.maxBrightness = 1024
    BG1.highlightSizeMaxBoost = 1024
    BG1.highlightSizePower = 1.5
    BG1.LoadFrom8bitRGB(r"resources/2DFrameExample_BG.png")
    BG1.ReconstructHighlight()
    BG1.UpdatePointSources()

    BG2 = Image2DFlatHighlightExtension()
    BG2.zDistance = 200000
    BG2.LoadFrom8bitRGB(r"resources/2DFrameExample_BGS.png")

    exampleStack = ImageStack()
    exampleStack.AddImage(BG2, "BG2")
    exampleStack.AddImage(BG1, "BG1")
    exampleStack.AddImage(MG, "MG")
    exampleStack.AddImage(FG, "FG")

    exampleStack.PrintLayerTags()

    return exampleStack


def ExampleStack2DNoGain():
    from .ImageExt import Image2DVariHighlightExtension, Image2DFlatHighlightExtension

    FG = Image2DFlatHighlightExtension()
    FG.zDistance = 900
    FG.LoadFrom8bitRGB(r"resources/2DFrameExample_FG.png")

    MG = Image2DVariHighlightExtension()
    MG.zDepthMappingRange = [1000, 1500]
    MG.LoadFrom8bitRGB(r"resources/2DFrameExample_MG.png")
    MG.LoadFrom8bitZ(r"resources/2DFrameExample_MGZ.png")
    #MG.ReconstructHighlight()
    MG.UpdatePointSources()

    BG1 = Image2DFlatHighlightExtension()
    BG1.zDistance = 35000
    #BG1.maxBrightness = 1024
    #BG1.highlightSizeMaxBoost = 1024
    #BG1.highlightSizePower = 1.5
    BG1.LoadFrom8bitRGB(r"resources/2DFrameExample_BG.png")
    #BG1.ReconstructHighlight()
    BG1.UpdatePointSources()

    BG2 = Image2DFlatHighlightExtension()
    BG2.zDistance = 200000
    BG2.LoadFrom8bitRGB(r"resources/2DFrameExample_BGS.png")

    exampleStack = ImageStack()
    exampleStack.AddImage(BG2, "BG2")
    exampleStack.AddImage(BG1, "BG1")
    exampleStack.AddImage(MG, "MG")
    exampleStack.AddImage(FG, "FG")

    exampleStack.PrintLayerTags()

    return exampleStack


def main():
    stack = ExampleStack2D()

if __name__ == "__main__":
    main()
