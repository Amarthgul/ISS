

import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR

from src.Raytracing.RayBatch import RayBatch
from .ImageVariDepth import Image2DVariDepth
from .Images import Image2D, Image2DFlat
from Util.Backend import backend as bd

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


    def PrintLayerTags(self):
        for index, key in enumerate(self.images.keys()):
            print(index, key)


    def EmitTowards(self, targets, sampleCount):

        wholeRB = RayBatch()

        for key, currentImage in self.images.items():
            wholeRB = currentImage.ReceiveAndEmitTowards(
                    targets,
                    wholeRB,
                    sampleCount)

        return wholeRB


def ExampleStack3D():

    FG = Image2DVariDepth()
    FG.zFarLimit = 1e3
    FG.LoadFromEXR(r"resources/DepthSceneFG.exr")

    # FG.DrawMask()

    MG = Image2DVariDepth()
    MG.LoadFromEXR(r"resources/DepthSceneMG.exr")
    print("MG Stats ======================")
    print(MG.Stats())

    MG2 = Image2DVariDepth()
    MG2.LoadFromEXR(r"resources/DepthSceneMG2.exr")
    print("MG2 Stats ======================")
    print(MG2.Stats())

    BG = Image2DVariDepth()
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