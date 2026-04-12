

import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.ImageIO import ImageConversion, SaveAsEXR
from Util.Misc import ProgressBar


class ImagingSystem:

    def __init__(self, lens, imager):

        self.lens = lens
        self.imager = imager

        self.object = None

        self.fNumber = 4

        self.focusDistance = 1500

        """Where the object is placed, this will likely never be used unless the target is a flat image."""
        self.objectDistance = 1500

        """Render time for each frame in second. If this is set then iteration will be ignored. This gives great control over render time but may produce uneven exposure among frames."""
        self.renderTime = 120

        """Number of iterations for each frame. This ensures exposure but may produce uneven time due to hardware performance fluctuation. """
        self.renderIteration = 32


        self._transmissionLoss = 0.8


    def RenderNamePattern(self, rex):

        pass


    def Render(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False):

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.BFD = self.lens.BestFocusBFD(focusDistance)
        self.imager.Update()

        image = self.imager.AcquireEmpty()

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))


        while (True):
            recorder = time.time()
            mainRB = self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(512), 20480)

            mainRB, mainRP, reflectedRB = self.lens.Propagate(mainRB, reflection=False)

            mainRB, _tir, _vig = self.imager.IntersectRays(mainRB)

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)

            if (realTimeUpdate):
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)


            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                # iterationCount
                image /= (iterationCount / self._transmissionLoss)

                fn = fileName
                SaveAsEXR(image, r"resources/Results", fn, flipHori=False, flipVert=True, rotate=True)

                break

            recorder = time.time()


    def _TerminatePercent(self, renderTime,  currentTime, renderIteration, currentIteration):
        # Prioritize passed in render time over others
        if renderTime is not None:
            return currentTime / renderTime

        # Prioritize time over iteration
        if self.renderTime is not None:
            return  currentTime / self.renderTime

        # Prioritize passed in iteration count over the build in one
        if renderIteration is not None:
            return currentIteration / renderIteration

        # Well, last resort
        if self.renderIteration is not None:
            return currentIteration / self.renderIteration


    def _TerminateCondition(self, renderTime, currentTime, renderIteration, currentIteration):

        return self._TerminatePercent(renderTime, currentTime, renderIteration, currentIteration) >= 1


