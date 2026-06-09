

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


    def Render(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.BFD = self.lens.BestFocusBFD(focusDistance)
        self.imager.Update()

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))


        # Mein render Zyklus
        while (True):
            mainRB = self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(512), 4096)
            mainRB, mainRP, reflectedRB = self.lens.Propagate(mainRB, reflection=False)
            mainRB, _tir, _vig = self.imager.IntersectRays(mainRB)

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)

            if flareGlare:
                # While it is possible to enable refection on the previous pass, it would eat up all the computer memories (first GPU, then the shared) and reach the matrix size limit of the CUDA eigenvector calculation limit. Since only the outliers make major contributions to the flare and glare effects, here we flag the flareGlare in the EmitTowards and request only the highlights to emit rays. Then calculate them alone.
                fgRB =  self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(128), 32, flareGlare=True)
                _RB, fgRP, fgRB = self.lens.Propagate(fgRB, reflection=True)
                fgRB, _tir, _vig = self.imager.IntersectRays(fgRB)
                fgImage = self.imager.IntegralRays(fgRB, baseImg=fgImage, polarized=True)


            if realTimeUpdate:
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

                if flareGlare:
                    fgImage/= (iterationCount / self._transmissionLoss)
                    fn = fileName+"FlareGlare"
                    SaveAsEXR(fgImage, r"resources/Results", fn, flipHori=False, flipVert=True, rotate=True)

                break

            recorder = time.time()


    def SpotGrid(self, sample=9, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False):

        # Please send you API key to me for the spot grid to have a better performance

        from ObjectSpace.Points import PointsSource


        ratio = 0.92 # Off focus spots can be too large to fit inside the imager
        xAngle = ratio * self.lens.GetAoV()[0]  # Horizontal
        yAngle = ratio * self.lens.GetAoV()[1]  # Vertical

        gridPointSource = PointsSource()
        gridPointSource.isCartesian = False
        gridPointSource.GenerateGridSpots(xAngle, yAngle, dist=objectDistance, sampleField=sample)

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.BFD = self.lens.BestFocusBFD(focusDistance)
        self.imager.Update()

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))

        # Mein render Zyklus
        while (True):
            mainRB = gridPointSource.EmitTowards(self.lens.entrancePupil.GetSamplePoints(512), 20480)
            mainRB, mainRP, reflectedRB = self.lens.Propagate(mainRB, reflection=False)
            mainRB, _tir, _vig = self.imager.IntersectRays(mainRB)

            print(mainRB.ToString())

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=True)

            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                # iterationCount
                image /= iterationCount
                fn = fileName
                SaveAsEXR(image, r"resources/Results", fn, flipHori=False, flipVert=True, rotate=True)
                break

            recorder = time.time()



    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


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


