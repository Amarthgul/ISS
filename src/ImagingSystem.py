

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


        self.sourcePerSample = 4096


        self.pupilPerSample = 512


    def Render(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            self.object,
            self.sourcePerSample
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
            mainRB = self.object.EmitTowards(self.lens.entrancePupil.GetSamplePoints(self.pupilPerSample), self.sourcePerSample)
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

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

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
                    )
                )
                break

            recorder = time.time()


    def RenderParaxial(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):
        """Render the primary image with centered Gaussian optics.

        The ideal path samples a finite front aperture, focuses at focusDistance
        in mm (default self.focusDistance), and leaves any separate flare pass physical.
        """
        distance = self.focusDistance if focusDistance is None else focusDistance
        self.imager.SetBFD(self.lens.ParaxialBFD(distance))
        self.imager.SetLensLength(self.lens.totalAxialLength)
        sampleTargets = self.lens.surfaces[0].SampleParaxialAperture
        matrices = self.lens.ParaxialSurfaceMatrices()
        self.imager.Update()

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

        image = self.imager.AcquireEmpty()
        fgImage = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            self.object,
            self.sourcePerSample
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
            mainRB = self.object.EmitTowards(sampleTargets(self.pupilPerSample), self.sourcePerSample)
            mainRB = self.lens.PropagateParaxial(mainRB, matrices)

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


    def SingleLayerAlpha(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):

        if self.singleObject is None:
            raise RuntimeError("ImagingSystem.SingleLayerAlpha(): self.singleObject cannot be None.")

        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

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
            self.sourcePerSample
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
            targets = self.lens.entrancePupil.GetSamplePoints(self.pupilPerSample)

            mainRB = self.singleObject.EmitTowards(targets, self.sourcePerSample)
            mainRB, _mainRP, _reflectedRB = self.lens.Propagate(mainRB, reflection=False)
            self.imager._AOVAverageCounts = beautyAOVAverageCounts
            image = self.imager.IntegralRays(mainRB, baseImg=image, polarized=False)
            beautyAOVAverageCounts = self.imager._AOVAverageCounts

            alphaRB = alphaStack.EmitTowards(targets, self.sourcePerSample, flareGlare=False)
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


    def SpotGrid(self, sample=9, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, ratio = 0.92, zonal=[0, 1]):

        # Please send you trading API key to me for the spot grid to have a better performance

        from ObjectSpace.Points import PointsSource

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

         # Off focus spots can be too large to fit inside the imager
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
            mainRB = gridPointSource.EmitTowards(self.lens.entrancePupil.GetSampleZonal(512, zonal[0], zonal[1]), 20480)
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

        if fNumber is not None:
            self.lens.SetAperture(fNumber)

        ISO12233C = Image2DFlat()
        ISO12233C.distance = objectDistance
        ISO12233C.horizontalAoV = self.lens.GetAoV()[0] * 2
        ISO12233C.LoadFrom8bit(r"resources/ISO12233-4k.png")

        image = self.imager.AcquireEmpty()

        sourceCoveragePerIteration = self._SourceCoveragePerIteration(
            ISO12233C,
            self.sourcePerSample
        )

        iterationCount = 0
        start = time.time()

        if (realTimeUpdate):
            plt.ion()  # Turn on interactive mode
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(image, flipH=True))

        # Mein render Zyklus
        while (True):
            mainRB = ISO12233C.ReceiveAndEmitTowards(self.lens.entrancePupil.GetSamplePoints(self.pupilPerSample), None, self.sourcePerSample)
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


    """
    T methods are all testing methods for specific and likely academic/scientific uses. 
    """

    def T_Polarization(self, objectDistance=None, focusDistance=None, fNumber=None, renderTime=None, iteration=None, fileName=None, realTimeUpdate=False, flareGlare=False):
        """Render a uniform white field and save I plus _Q/_U/_AoLP/_DoLP.

        Stokes powers accumulate before deriving polarization. Raw AoLP radians
        and DoLP fractions pass through the imager's output filters before
        saving. The primary normalized/filtered maps are retained on the imager.
        Optional flare/glare maps use the additional FlareGlare filename suffix.
        """
        from ObjectSpace.Images import EmptyImageSource

        ratio = 1.1
        distance = self.objectDistance if objectDistance is None else objectDistance
        self.imager.SetLensLength(self.lens.totalAxialLength)
        self.imager.SetBFD(self.lens.BestFocusBFD(focusDistance))
        self.imager.Update()
        if fNumber is not None:
            self.lens.SetAperture(fNumber)

        angles = self.lens.GetAoV(unitInDegrees=False,
                                  w=self.imager.width, h=self.imager.height)
        halfSize = ratio * distance * bd.tan(angles[:2])
        source = EmptyImageSource(
            distance, halfSize[0], halfSize[1],
            dimension=[self.imager.horizontalPx, self.imager.verticalPx])

        # Use a small separate reflection pass, as in the ordinary renderer.
        passes = [(False, self.pupilPerSample, self.sourcePerSample, "")]
        if flareGlare:
            passes.append((True, 8, 8, "FlareGlare"))
        accumulated = [[self.imager.AcquireEmpty() for _ in range(3)] for _ in passes]
        coverage = [self._SourceCoveragePerIteration(source, count)
                    for _, _, count, _ in passes]

        iterationCount = 0
        start = time.time()
        if realTimeUpdate:
            plt.ion()
            fig, ax = plt.subplots()
            im = ax.imshow(ImageConversion(accumulated[0][0], flipH=True))

        while True:
            for (reflection, pupilCount, sourceCount, _), maps in zip(passes, accumulated):
                rays = source.EmitTowards(
                    self.lens.entrancePupil.GetSamplePoints(pupilCount), sourceCount)
                mainRB, _rayPath, reflectedRB = self.lens.Propagate(rays, reflection=reflection)
                detected = reflectedRB if reflection else mainRB
                maps[0] += self.imager.IntegralPolarization(detected, calculateDerived=False)
                maps[1] += self.imager.Q
                maps[2] += self.imager.U

            if realTimeUpdate:
                im.set_data(ImageConversion(accumulated[0][0], flipV=True, maxModifier=0.1))
                plt.draw()
                plt.pause(0.01)

            elapsed = time.time() - start
            iterationCount += 1
            ProgressBar(self._TerminatePercent(renderTime, elapsed, iteration, iterationCount), 100)
            if self._TerminateCondition(renderTime, elapsed, iteration, iterationCount):
                break

        for (_, _, _, suffix), (intensity, q, u), sourceCoverage in zip(passes, accumulated, coverage):
            scalar = self._transmissionLoss / (iterationCount * sourceCoverage)
            intensity *= scalar
            q *= scalar
            u *= scalar
            linear = bd.hypot(q, u)
            dolp = linear / bd.where(intensity > 0, intensity, bd.nan)
            aolp = bd.where((intensity > 0) & (linear > 0),
                            0.5 * bd.arctan2(u, q), bd.nan)
            aolp = self.imager.FilterAoLP(aolp)
            dolp = self.imager.FilterDoLP(dolp)
            for name, values in (("", intensity), ("_Q", q), ("_U", u),
                                 ("_AoLP", aolp), ("_DoLP", dolp)):
                self.imager.SaveImage(fileName + suffix + name, sourceImg=values)
            if not suffix:
                self.imager.image = intensity
                self.imager.Q, self.imager.U = q, u
                self.imager.AoLP, self.imager.DoLP = aolp, dolp

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


