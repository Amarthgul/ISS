

import time
import copy
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.ImageIO import ImageConversion
from Util.Misc import ProgressBar
from ObjectSpace.ImageStack import ImageStack


class ImagingSystem:

    def __init__(self, lens, imager):

        self.lens = lens
        self.imager = imager

        self.object = None

        self.singleObject = None

        self.fNumber = 4

        self.focusDistance = 1500

        """Where the object is placed, this will likely never be used unless the target is a flat image."""
        self.objectDistance = 1500

        """Render time for each frame in second. If this is set then iteration will be ignored. This gives great control over render time but may produce uneven exposure among frames."""
        self.renderTime = 120

        """Number of iterations for each frame. This ensures exposure but may produce uneven time due to hardware performance fluctuation. """
        self.renderIteration = 32

        """Most system does not have 100% transmission. Direct image will be darker, so this is the transmission compensator."""
        self._transmissionLoss = 0.8

        self._sampleMagicNumber= 2 ** 10
        # I have no idea why this is needed, the sample ratio should have normalized the output image brightness, but still, a 10-stop compensator is required.


        self._sourcePerSample = 20480


        self._pupilPerSample = 512


    def RenderNamePattern(self, rex):

        pass


    def Render(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            self.object,
            self._sourcePerSample
        )
        flareCoveragePerIteration = None
        if flareGlare:
            flareCoveragePerIteration = self._SourceCoveragePerIteration(
                self.object,
                8,
                highlightSources=True
            )

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))


        # Mein render Zyklus
        while (True):
            mainRB = self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(self._pupilPerSample), self._sourcePerSample)
            mainRB, mainRP, reflectedRB = self.lens.Propagate(mainRB, reflection=False)

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)

            if flareGlare:
                # While it is possible to enable refection on the previous pass, it would eat up all the computer memories (first GPU, then the shared) and reach the matrix size limit of the CUDA eigenvector calculation limit. Since only the outliers make major contributions to the flare and glare effects, here we flag the flareGlare in the EmitTowards and request only the highlights to emit rays. Then calculate them alone.
                fgRB =  self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(8), 8, flareGlare=True)
                _RB, fgRP, fgRB = self.lens.Propagate(fgRB, reflection=True)
                fgImage = self.imager.IntegralRays(fgRB, baseImg=fgImage, polarized=True)


            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)


            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                self.imager.SaveImage(
                    fileName,
                    sourceImg=image,
                    scalar=(
                        self._transmissionLoss
                        / (iterationCount * sourceCoveragePerIteration)
                    )
                )

                if flareGlare:
                    self.imager.SaveImage(
                        fileName + "FlareGlare",
                        sourceImg=fgImage,
                        scalar=(
                            self._transmissionLoss
                            / (iterationCount * flareCoveragePerIteration)
                        )
                    )

                break

            recorder = time.time()


    def RenderFlareOnly(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):
        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        fgImage = self.imager.AcquireEmpty()

        flareCoveragePerIteration = self._SourceCoveragePerIteration(
            self.object,
            4,
            highlightSources=True
        )

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(fgImage, flipH=True))

        # Mein render Zyklus
        while (True):
            # Only render the flare/glare reflection pass; skip the sequential image pass entirely.
            fgRB = self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(8), 4, flareGlare=True)
            _RB, fgRP, fgRB = self.lens.Propagate(fgRB, reflection=True)
            fgImage = self.imager.IntegralRays(fgRB, baseImg=fgImage, polarized=True)

            if realTimeUpdate:
                im.set_data(ImageConversion(fgImage, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                self.imager.SaveImage(
                    fileName + "FlareGlare",
                    sourceImg=fgImage,
                    scalar=(
                        self._transmissionLoss
                        / (iterationCount * flareCoveragePerIteration)
                    )
                )
                break

            recorder = time.time()


    def SingleLayerAlpha(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):

        if self.singleObject is None:
            raise RuntimeError("ImagingSystem.SingleLayerAlpha(): self.singleObject cannot be None.")

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        alphaChannelName = "X"
        alphaStack = ImageStack()
        alphaStack.AlphaImage(copy.deepcopy(self.singleObject), alphaChannelName)
        alphaAOVNames = alphaStack.GetAOVNames()
        if alphaChannelName not in alphaAOVNames:
            raise RuntimeError(
                f"ImagingSystem.SingleLayerAlpha(): AlphaImage did not create AOV '{alphaChannelName}'."
            )
        alphaChannelIndex = alphaAOVNames.index(alphaChannelName)

        image = self.imager.AcquireEmpty()
        alphaImage = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            self.singleObject,
            self._sourcePerSample
        )
        flareCoveragePerIteration = None
        if flareGlare:
            flareCoveragePerIteration = self._SourceCoveragePerIteration(
                self.singleObject,
                8,
                highlightSources=True
            )

        previousAOVAverageCounts = getattr(self.imager, "_AOVAverageCounts", None)
        beautyAOVAverageCounts = None
        alphaAOVAverageCounts = None
        flareAOVAverageCounts = None

        iterationCount = 0
        start = time.time()

        if realTimeUpdate:
            plt.ion()
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))

        while True:
            targets = self.lens.entrancePupil.GetSamplePoints(self._pupilPerSample)

            mainRB = self.singleObject.EmitTowards(targets, self._sourcePerSample)
            mainRB, _mainRP, _reflectedRB = self.lens.Propagate(mainRB, reflection=False)
            self.imager._AOVAverageCounts = beautyAOVAverageCounts
            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)
            beautyAOVAverageCounts = self.imager._AOVAverageCounts

            alphaRB = alphaStack.EmitTowards(targets, self._sourcePerSample, flareGlare=False)
            alphaRB, _alphaRP, _alphaReflectedRB = self.lens.Propagate(alphaRB, reflection=False)
            self.imager._AOVAverageCounts = alphaAOVAverageCounts
            alphaImage = self.imager.IntegralRays(alphaRB, baseImg=alphaImage, polarized=False)
            alphaAOVAverageCounts = self.imager._AOVAverageCounts

            if flareGlare:
                fgRB = self.singleObject.EmitTowards(self.lens.entrancePupil.GetSamplePoints(8), 8, flareGlare=True)
                _RB, fgRP, fgRB = self.lens.Propagate(fgRB, reflection=True)
                self.imager._AOVAverageCounts = flareAOVAverageCounts
                fgImage = self.imager.IntegralRays(fgRB, baseImg=fgImage, polarized=True)
                flareAOVAverageCounts = self.imager._AOVAverageCounts

            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                imageScalar = self._transmissionLoss / (iterationCount * sourceCoveragePerIteration)

                alphaImageChannel = 3 + alphaChannelIndex
                if alphaImage.shape[-1] <= alphaImageChannel:
                    raise RuntimeError(
                        "ImagingSystem.SingleLayerAlpha(): alpha branch did not produce the AlphaImage AOV channel."
                    )

                alpha = bd.clip(1.0 - alphaImage[:, :, alphaImageChannel], 0.0, 1.0)

                self.imager.SaveImage(
                    fileName,
                    sourceImg=image[:, :, :3],
                    scalar=imageScalar,
                    extraChannels=(alpha, "A")
                )

                if flareGlare:
                    self.imager.SaveImage(
                        fileName + "FlareGlare",
                        sourceImg=fgImage,
                        scalar=(
                            self._transmissionLoss
                            / (iterationCount * flareCoveragePerIteration)
                        )
                    )

                self.imager._AOVAverageCounts = previousAOVAverageCounts
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
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            gridPointSource,
            20480
        )

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

            # print(mainRB.ToString())

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=True)

            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                self.imager.SaveImage(
                    fileName,
                    sourceImg=image,
                    scalar=1 / (iterationCount * sourceCoveragePerIteration)
                )
                break

            recorder = time.time()


    def ISO12233(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False):
        from ObjectSpace.Image2DFlat import Image2DFlat

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        ISO12233C = Image2DFlat()
        ISO12233C.distance = objectDistance
        ISO12233C.horizontalAoV = self.lens.GetAoV()[0] * 2
        ISO12233C.LoadFrom8bit(r"resources/ISO12233-4k.png")

        image = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            ISO12233C,
            self._sourcePerSample
        )

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))

        # Mein render Zyklus
        while (True):
            mainRB = ISO12233C.ReceiveAndEmitTowards( self.lens.entrancePupil.GetSamplePoints(self._pupilPerSample), None,  self._sourcePerSample)
            mainRB, mainRP, reflectedRB = self.lens.Propagate(mainRB, reflection=False)

            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)

            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                self.imager.SaveImage(
                    fileName,
                    sourceImg=image,
                    scalar=(
                        self._transmissionLoss
                        / (iterationCount * sourceCoveragePerIteration)
                    )
                )

                break

            recorder = time.time()


    def T_OutlierRender(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):
        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        image = self.imager.AcquireEmpty()

        highlightSampleCount = 4
        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            self.object,
            highlightSampleCount,
            highlightSources=True
        )

        iterationCount = 0
        start = time.time()

        if realTimeUpdate:
            plt.ion()
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))

        while True:
            mainRB = self.object.EmitTowards(
                self.lens.entrancePupil.GetSamplePoints(8),
                highlightSampleCount,
                flareGlare=True
            )
            mainRB, _mainRP, _reflectedRB = self.lens.Propagate(
                mainRB,
                reflection=False
            )
            image = self.imager.IntegralRays(
                mainRB,
                baseImg=image,
                polarized=False
            )

            if realTimeUpdate:
                im.set_data(ImageConversion(image, flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1

            ProgressBar(
                self._TerminatePercent(renderTime, elapsed, iteration, iterationCount),
                100
            )

            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                self.imager.SaveImage(
                    fileName,
                    sourceImg=image,
                    scalar=(
                        self._transmissionLoss
                        / (iterationCount * sourceCoveragePerIteration)
                    )
                )
                break


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _SourceCoveragePerIteration(self, source, sampleCount, highlightSources=False):
        """Return the fraction of the source population sampled by one iteration.

        Each image in an ``ImageStack`` is sampled independently, so stacked
        populations are added before calculating the coverage.  A sample count
        larger than a population is capped at that population, matching
        ``PointsSource.EmitSamplesToward``.

        A coverage of 1 is returned for an empty source.  Its accumulated image
        is black, and retaining the legacy per-iteration normalization avoids a
        divide-by-zero without changing the result.
        """
        if sampleCount <= 0:
            raise ValueError("ImagingSystem source sample count must be greater than zero.")

        if isinstance(source, ImageStack):
            sourceObjects = source.images.values()
        else:
            sourceObjects = (source,)

        totalSourceCount = 0
        sampledSourceCount = 0

        for sourceObject in sourceObjects:
            pointSource = None

            if highlightSources and hasattr(sourceObject, "pointSourceHigh"):
                pointSource = sourceObject.pointSourceHigh
                if pointSource is None and hasattr(sourceObject, "ConstructHighlightPoints"):
                    pointSource = sourceObject.ConstructHighlightPoints()
            elif hasattr(sourceObject, "pointSource"):
                pointSource = sourceObject.pointSource
            elif hasattr(sourceObject, "value"):
                # PointsSource can be rendered directly.
                pointSource = sourceObject

            values = pointSource.value
            if values is None:
                continue

            currentSourceCount = int(values.shape[0])
            totalSourceCount += currentSourceCount
            sampledSourceCount += min(int(sampleCount), currentSourceCount)

        if totalSourceCount == 0:
            return 1.0

        return (sampledSourceCount / totalSourceCount) * self._sampleMagicNumber


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


