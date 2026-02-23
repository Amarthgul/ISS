

import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR

from src.Raytracing.RayBatch import RayBatch
from .ImageVariDepth import Image2DVariDepth
from .Images import Image2D, Image2DFlat


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


def ExampleStack():

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


def main():
    stack = ExampleStack()

if __name__ == "__main__":
    main()