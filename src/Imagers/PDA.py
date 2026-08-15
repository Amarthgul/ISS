
from .Standard import StdImager
from Material import Material
from Surfaces.Surface import Surface
from Util.Globals import INFINITY, ZERO, ONE, RNG
from Util.Backend import backend as bd
from Util.ColorPDF import ColorPDF
from Util.ImageIO import SaveFormat, SaveAsDNG, SaveAsEXR





class PDA(StdImager):
    """
    Photodiode Array. 
    This class refers to both CMOS and CCD. 
    """

    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):
        super().__init__(bfd, w, h, horiPx)

        """This is the thickness of the UVIR cut glass"""
        self.tUVIR = 1.5

        """This is the distance between the UVIR glass and the sensor plane"""
        self.t = .5

        self.material = "UVIR"

        self.surfaces = []


        """Pixel fill factor, defaults to the ideal 1, i.e., 100%."""
        self.fillFactor = 1

        self.pixelPitch = None


        """When enabled, UVIR will be calculated by the imager.
        This makes things easier with a minor cost that flares and glares caused by UVIR becomes harder to represent."""
        self.selfResponsibleUVIR = True


        # MLA and PDA are abstracted to have 0 thickness and occupies the same plane
        """Depth from the MLA/CFA to the actual PDA plane."""
        self.wellDepth = 0.02


        # ==================================================================
        """ ===================== Micro Lens Array ===================== """
        # ==================================================================

        """The material used for MLA. By default a composite with 1.66 nd and 20.4 Vd."""
        self.materialMLA = Material("OKP4")


        """For some sensor designs, Leica especially, the MLA are shifted laterally to accommodate sharp incident angle."""
        self.lateralScale = 1


        self._singleLensNormal = None
        self._normalMapSize = 64
        self._lensRadiusModifier = 1

        # ==================================================================
        """ =================== Color Filtering Array =================== """
        # ==================================================================

        self.CFAPattern = "RGGB"

        """Spectral response of the red, green, and blue CFA filters."""
        self.CFAResponse = ColorPDF()
        self._cfaPatternMap = None

        # ==================================================================
        """ ======================= Image output ======================= """
        # ==================================================================

        self.outputFormat = SaveFormat.DNG
        self.dngMetadata = {
            "make": "ISS",
            "model": "Virtual PDA",
            "uniqueCameraModel": "ISS Virtual PDA",
            "software": "ISS Optical Simulation",
            "colorMatrix1": (
                3.2404542, -1.5371385, -0.4985314,
                -0.9692660, 1.8760108, 0.0415560,
                0.0556434, -0.2040259, 1.0572252
            ),
            "asShotNeutral": (1.0, 1.0, 1.0),
            "calibrationIlluminant1": 21,
            "baselineExposure": 0.0,
            "orientation": 1
        }

        """Optional lens and exposure information written to DNG metadata."""
        self.lensModel = None
        self.ISO = None
        self.focalLength = None
        self.aperture = None

        self.rawImage = None

        """When enabled, an EXR image will also be saved along side the DNG"""
        self.backupDNG = False

        self.rawBitDepth = 14
        self.blackLevel = 0
        self.whiteLevel = (1 << self.rawBitDepth) - 1


    def GetUVIR(self):
        """
        Acquire the surfaces of the UVIR glass.
        """

        return self.surfaces


    def SetBFD(self, BFD):
        """
        Set the active PDA plane from the lens-only, on-axis best focus.

        A plane-parallel cover glass shifts the focus away from the lens by ``t * (1 - 1 / n)``.  The active photodiode plane is then a further ``wellDepth`` behind the MLA/CFA plane.
        """
        n_d = Material(self.material).n_d()
        glassFocusShift = self.tUVIR * (1.0 - 1.0 / n_d)

        super().SetBFD(BFD + glassFocusShift + self.wellDepth)


    def Update(self):
        """
        This should be called last.
        """
        self.pixelPitch = self.width / self.horizontalPx
        self._ValidateConfiguration()

        # Sensor parameters remain fixed during a render, so build the MLA
        # lookup once here instead of checking or regenerating it per batch.
        self._GenerateSingleLensNormal()

        diagonal = bd.sqrt((self.width / 2.0) ** 2 + (self.height / 2.0) ** 2) + 5

        self.surfaces = [
            Surface(INFINITY, self.tUVIR, diagonal, self.material),
            Surface(INFINITY, self.t, diagonal)
        ]

        super().Update()

        # The imager position is the active PDA plane.
        # Work backwards through the well depth and air gap to position the rear and front UVIR faces.
        rearUVIRPosition = self._zPos - self.wellDepth - self.t
        frontUVIRPosition = rearUVIRPosition - self.tUVIR

        self.surfaces[0].SetCumulative(frontUVIRPosition)
        self.surfaces[1].SetCumulative(rearUVIRPosition)


    def IntegralRays(self, raybatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):

        if self.selfResponsibleUVIR:
            raybatch = self._UVIRRefract(raybatch)

        raybatch = self._ApplyMLAAndCFA(raybatch)

        return self._PDAIntegral(raybatch, baseImg, overExpNoiseRemoval, polarized)


    def SaveImage(self, fileName, sourceImg=None, scalar=1, extraChannels=()):
        """Save the accumulated sensor image as EXR, DNG, or DNG plus EXR."""
        image = self.image if sourceImg is None else sourceImg
        if image is None:
            raise ValueError("PDA.SaveImage(): no image is available to save.")

        outputFormat = self.outputFormat
        if isinstance(outputFormat, str):
            try:
                outputFormat = SaveFormat[outputFormat.upper()]
            except KeyError as error:
                raise ValueError(f"Unsupported PDA output format: {self.outputFormat}") from error

        scaledImage = bd.asarray(image) * scalar
        outputFolder = r"resources/Results"

        if outputFormat == SaveFormat.EXR:
            SaveAsEXR(
                scaledImage,
                outputFolder,
                fileName,
                *extraChannels,
                flipHori=False,
                flipVert=True,
                rotate=True
            )
            return

        if outputFormat != SaveFormat.DNG:
            raise ValueError(f"Unsupported PDA output format: {self.outputFormat}")

        self.rawImage = self._BuildBayerMosaic(scaledImage)
        metadata = dict(self.dngMetadata)
        optionalMetadata = {
            "lensModel": self.lensModel,
            "iso": self.ISO,
            "focalLength": self.focalLength,
            "aperture": self.aperture
        }
        metadata.update(
            (key, value)
            for key, value in optionalMetadata.items()
            if value is not None
        )
        SaveAsDNG(
            self.rawImage,
            outputFolder,
            fileName,
            cfaPattern=self.CFAPattern,
            bitDepth=self.rawBitDepth,
            blackLevel=self.blackLevel,
            whiteLevel=self.whiteLevel,
            metadata=metadata
        )

        # AOVs have no direct home in the raw DNG.  When requested, preserve
        # the sparse RGB channels and any explicit AOVs in a companion EXR.
        if self.backupDNG:
            SaveAsEXR(
                scaledImage,
                outputFolder,
                fileName,
                *extraChannels,
                flipHori=False,
                flipVert=True,
                rotate=True
            )


    def RunTest(self):
        self._GenerateSingleLensNormal()
        self._PlotNormal()


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _UVIRRefract(self, raybatch):
        """
        Propagate rays through the front and rear surfaces of the UVIR glass.

        The incoming rays are in air.  Each surface then supplies the material
        on its image side, matching the refractive-index convention used by
        :meth:`Surface.Trace`.
        """
        if raybatch is None or raybatch.IsNoneType():
            return raybatch

        if len(self.surfaces) != 2:
            raise RuntimeError(
                "PDA.Update() must create the two UVIR surfaces before "
                "PDA._UVIRRefract() is called."
            )

        previousRI = bd.ones_like(raybatch.Wavelength()) # Air

        for surface in self.surfaces:
            raybatch, _tir, _vig, _stray = surface.Trace(
                raybatch,
                previousRI
            )

            if raybatch.IsNoneType():
                return raybatch

            previousRI = surface.RI(raybatch.Wavelength())

        return raybatch


    def _ApplyMLAAndCFA(self, raybatch):
        """
        Calculate the ray intersections at the MLA/CFA plane and how they modulate the ray.
        """
        if raybatch is None or raybatch.IsNoneType():
            return raybatch

        mlaPosition = self._zPos - self.wellDepth
        raybatch = self._IntersectPlane(raybatch, mlaPosition)
        if raybatch.IsNoneType():
            return raybatch

        pixelX, pixelY, uv, inBounds = self._PixelCoordinates(raybatch.Position())
        raybatch.Mask(inBounds)
        if raybatch.IsNoneType():
            return raybatch

        pixelX = pixelX[inBounds]
        pixelY = pixelY[inBounds]
        uv = uv[inBounds]

        filterChannels = self._CFAChannels(pixelX, pixelY)
        incomingChannels = raybatch.Channel().astype(bd.int32)

        # A ray already marked as the filter's color passes unconditionally.
        # Cross-channel light is admitted stochastically according to that
        # filter's spectral response at the ray wavelength.
        crossChannelResponse = self.CFAResponse.SpectralResponse(
            raybatch.Wavelength(),
            filterChannels
        )
        acceptanceProbability = bd.where(
            incomingChannels == filterChannels,
            ONE,
            bd.clip(crossChannelResponse, ZERO, ONE)
        )
        accepted = RNG.rand(raybatch.value.shape[0]) < acceptanceProbability

        raybatch.Mask(accepted)
        if raybatch.IsNoneType():
            return raybatch

        uv = uv[accepted]
        filterChannels = filterChannels[accepted]

        # Once through the CFA, the ray belongs to the filter/well channel.
        raybatch.SetChannel(filterChannels)

        normals = self._SampleMLANormal(uv)
        raybatch = self._ApplyMLARefraction(raybatch, normals)

        return raybatch


    def _PDAIntegral(self, raybatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """Propagate rays to the active wells and integrate their charge."""
        if raybatch is not None and not raybatch.IsNoneType():
            raybatch = self._IntersectPlane(raybatch, self._zPos)

        if raybatch is not None and not raybatch.IsNoneType():
            _pixelX, _pixelY, wellUV, inBounds = self._WellCoordinates(raybatch.Position())

            if self.fillFactor <= 0:
                activeWell = bd.zeros_like(inBounds)
            elif self.fillFactor >= 1:
                activeWell = inBounds
            else:
                activeHalfWidth = bd.sqrt(self.fillFactor) / 2.0
                activeWell = inBounds & bd.all(
                    bd.abs(wellUV - 0.5) <= activeHalfWidth,
                    axis=1
                )

            raybatch.Mask(activeWell)

        if raybatch is None or raybatch.value is None:
            image = self.AcquireEmpty() if baseImg is None else bd.asarray(baseImg)
        else:
            image = super()._integralRaysChannelBased(
                raybatch,
                baseImg=baseImg,
                overExpNoiseRemoval=overExpNoiseRemoval,
                polarized=polarized
            )

        self.image = image
        self.rawImage = self._BuildBayerMosaic(image)
        return image


    def _IntersectPlane(self, raybatch, zPosition):
        """Move forward-facing rays to an axial plane and discard misses."""
        if raybatch is None or raybatch.IsNoneType():
            return raybatch

        positions = raybatch.Position()
        directions = raybatch.Direction()
        directionZ = directions[:, 2]

        parallel = bd.isclose(directionZ, ZERO)
        safeDirectionZ = bd.where(parallel, ONE, directionZ)
        distance = (zPosition - positions[:, 2]) / safeDirectionZ

        valid = (~parallel) & bd.isfinite(distance) & (distance >= ZERO)
        intersections = positions + distance[:, bd.newaxis] * directions

        raybatch.Mask(valid)
        if not raybatch.IsNoneType():
            raybatch.SetPosition(intersections[valid])

        return raybatch


    def _PixelCoordinates(self, intersections):
        """Map intersections to well indices and shifted microlens-local UVs."""
        pixelOffset = bd.array([
            self.horizontalPx / 2.0,
            self.verticalPx / 2.0
        ])
        continuous = intersections[:, :2] / self.pixelPitch + pixelOffset
        pixelIndices = bd.floor(continuous).astype(bd.int32)

        # CFA/well indices remain on the regular sensor grid. MLA centers can
        # be scaled radially toward the sensor center, making edge microlenses
        # sit slightly inward from the wells beneath them. A scale of 1 keeps
        # the MLA and well centers aligned.
        wellCenters = (pixelIndices + 0.5 - pixelOffset) * self.pixelPitch
        mlaCenters = wellCenters * self.lateralScale
        uv = (intersections[:, :2] - mlaCenters) / self.pixelPitch + 0.5

        pixelX = pixelIndices[:, 0]
        pixelY = pixelIndices[:, 1]
        inBounds = (
            (pixelX >= 0) & (pixelX < self.horizontalPx) &
            (pixelY >= 0) & (pixelY < self.verticalPx)
        )

        return pixelX, pixelY, uv, inBounds


    def _WellCoordinates(self, intersections):
        """Map intersections to active-well indices and unshifted local UVs."""
        pixelOffset = bd.array([
            self.horizontalPx / 2.0,
            self.verticalPx / 2.0
        ])
        continuous = intersections[:, :2] / self.pixelPitch + pixelOffset
        pixelIndices = bd.floor(continuous).astype(bd.int32)
        wellUV = continuous - pixelIndices

        pixelX = pixelIndices[:, 0]
        pixelY = pixelIndices[:, 1]
        inBounds = (
            (pixelX >= 0) & (pixelX < self.horizontalPx) &
            (pixelY >= 0) & (pixelY < self.verticalPx)
        )

        return pixelX, pixelY, wellUV, inBounds


    def _BuildBayerMosaic(self, image):
        """Collapse sparse RGB sensor channels into a row-major raw mosaic."""
        image = bd.asarray(image)
        if image.ndim == 2:
            mosaic = image
        elif image.ndim == 3 and image.shape[-1] >= 3:
            mosaic = bd.sum(image[:, :, :3], axis=2)
        else:
            raise ValueError(
                "PDA Bayer conversion expects a 2D raw image or an image "
                "with at least three channels."
            )

        # Internal images use (pixelX, pixelY); TIFF/DNG uses (rowY, columnX).
        return bd.transpose(mosaic, (1, 0))


    def _SampleMLANormal(self, uv):
        """Bilinearly sample and normalize the single-microlens normal map."""
        if self._singleLensNormal is None:
            self._GenerateSingleLensNormal()

        normalMap = self._singleLensNormal
        height, width = normalMap.shape[:2]

        uv = bd.clip(uv, ZERO, ONE)
        x = uv[:, 0] * (width - 1)
        y = uv[:, 1] * (height - 1)

        x0 = bd.floor(x).astype(bd.int32)
        y0 = bd.floor(y).astype(bd.int32)
        x1 = bd.minimum(x0 + 1, width - 1)
        y1 = bd.minimum(y0 + 1, height - 1)

        weightX = (x - x0)[:, bd.newaxis]
        weightY = (y - y0)[:, bd.newaxis]

        top = normalMap[y0, x0] * (ONE - weightX) + normalMap[y0, x1] * weightX
        bottom = normalMap[y1, x0] * (ONE - weightX) + normalMap[y1, x1] * weightX
        normals = top * (ONE - weightY) + bottom * weightY

        magnitudes = bd.linalg.norm(normals, axis=1, keepdims=True)
        return normals / bd.maximum(magnitudes, 1e-12)


    def _ApplyMLARefraction(self, raybatch, normals):
        """Apply one effective air-to-MLA refraction at the sampled normals."""
        if raybatch is None or raybatch.IsNoneType():
            return raybatch

        directions = raybatch.Direction()
        directions /= bd.maximum(
            bd.linalg.norm(directions, axis=1, keepdims=True),
            1e-12
        )

        # The stored normal map uses +Z as its neutral tangent-space normal.
        # Rays approach the sensor along +Z, so the physical entrance normal has the same transverse components and faces toward -Z.
        surfaceNormals = bd.copy(normals)
        surfaceNormals[:, 2] = -bd.abs(surfaceNormals[:, 2])

        wrongFacing = bd.sum(directions * surfaceNormals, axis=1) > ZERO
        surfaceNormals = bd.where(
            wrongFacing[:, bd.newaxis],
            -surfaceNormals,
            surfaceNormals
        )

        incidentRI = bd.ones_like(raybatch.Wavelength())
        mlaRI = self.materialMLA.RI(raybatch.Wavelength())
        ratio = incidentRI / mlaRI

        cosineIncident = -bd.sum(directions * surfaceNormals, axis=1)
        discriminant = ONE - ratio ** 2 * (ONE - cosineIncident ** 2)
        transmitted = discriminant >= ZERO

        factor = ratio * cosineIncident - bd.sqrt(bd.maximum(discriminant, ZERO))
        refracted = (
            ratio[:, bd.newaxis] * directions +
            factor[:, bd.newaxis] * surfaceNormals
        )
        refracted /= bd.maximum(
            bd.linalg.norm(refracted, axis=1, keepdims=True),
            1e-12
        )

        # Entry from air into the MLA should not ordinarily produce TIR, but sufficiently unusual materials or incidence angles can still make it occur.
        # Such rays do not reach the photodiode.
        raybatch.Mask(transmitted)
        if not raybatch.IsNoneType():
            raybatch.SetDirection(refracted[transmitted])

        return raybatch


    def _CFAChannels(self, pixelX, pixelY):
        """Return cached RGB channel IDs for the configured Bayer pattern."""
        return self._cfaPatternMap[pixelY % 2, pixelX % 2]


    def _ValidateConfiguration(self):
        """Validate render-invariant PDA settings and cache lookup values."""
        pattern = self.CFAPattern.upper()
        if (
            len(pattern) != 4 or
            pattern.count("R") != 1 or
            pattern.count("G") != 2 or
            pattern.count("B") != 1
        ):
            raise ValueError(
                "PDA.CFAPattern must be a 2x2 Bayer pattern such as "
                "'RGGB', 'BGGR', 'GRBG', or 'GBRG'."
            )

        if not isinstance(self.CFAResponse, ColorPDF):
            raise ValueError("PDA.CFAResponse must be a ColorPDF instance.")
        self.CFAResponse._Update()

        lateralScale = self.lateralScale
        if hasattr(lateralScale, "get"):
            lateralScale = lateralScale.get()
        lateralScale = float(lateralScale)
        if lateralScale <= 0.0 or lateralScale > 1.0:
            raise ValueError("PDA.lateralScale must be greater than 0 and no greater than 1.")

        if int(self._normalMapSize) < 2:
            raise ValueError("PDA._normalMapSize must be at least 2.")
        if self._lensRadiusModifier <= 0:
            raise ValueError("PDA._lensRadiusModifier must be greater than zero.")
        if self.fillFactor < 0 or self.fillFactor > 1:
            raise ValueError("PDA.fillFactor must be between 0 and 1.")

        outputFormat = self.outputFormat
        if isinstance(outputFormat, str):
            try:
                outputFormat = SaveFormat[outputFormat.upper()]
            except KeyError as error:
                raise ValueError(f"Unsupported PDA output format: {self.outputFormat}") from error
        if not isinstance(outputFormat, SaveFormat):
            raise ValueError(f"Unsupported PDA output format: {self.outputFormat}")

        if int(self.rawBitDepth) < 8 or int(self.rawBitDepth) > 16:
            raise ValueError("PDA.rawBitDepth must be between 8 and 16.")
        if self.blackLevel < 0 or self.whiteLevel <= self.blackLevel or self.whiteLevel > 65535:
            raise ValueError(
                "PDA raw levels must satisfy 0 <= blackLevel < whiteLevel <= 65535."
            )
        if not isinstance(self.dngMetadata, dict):
            raise ValueError("PDA.dngMetadata must be a dictionary.")

        channelID = {"R": 0, "G": 1, "B": 2}
        self._cfaPatternMap = bd.array([
            [channelID[pattern[0]], channelID[pattern[1]]],
            [channelID[pattern[2]], channelID[pattern[3]]]
        ], dtype=bd.int32)



    def _GenerateSingleLensNormal(self):
        """Generate the spherical normal map for one square microlens."""
        if self.pixelPitch is None:
            raise RuntimeError("PDA.Update() must be called before generating the microlens normal map.")

        mapSize = int(self._normalMapSize)

        halfPitch = self.pixelPitch / 2.0
        pixelDiagonal = bd.sqrt(2.0) * self.pixelPitch
        sphereDiameter = self._lensRadiusModifier * pixelDiagonal
        sphereRadius = sphereDiameter / 2.0

        coordinates = bd.linspace(-halfPitch, halfPitch, mapSize)
        xx, yy = bd.meshgrid(coordinates, coordinates, indexing="xy")

        # Select the positive spherical branch. Geometrically, this places the sphere center underneath the pixel plane and exposes its convex side.
        radialSquared = xx ** 2 + yy ** 2
        hemisphereMask = radialSquared <= sphereRadius ** 2
        zz = bd.sqrt(bd.maximum(sphereRadius ** 2 - radialSquared, ZERO))
        normalMap = bd.stack((xx, yy, zz), axis=2)

        magnitudes = bd.linalg.norm(normalMap, axis=2, keepdims=True)
        normalMap = normalMap / bd.maximum(magnitudes, 1e-12)

        # A hemisphere smaller than its pixel leaves part of the square map uncovered. Keep that region optically neutral rather than using horizontal sphere normals.
        upwardNormal = bd.array([ZERO, ZERO, ONE])
        self._singleLensNormal = bd.where(
            hemisphereMask[:, :, bd.newaxis],
            normalMap,
            upwardNormal
        )

        return self._singleLensNormal


    def _PlotNormal(self):
        """Display the single-microlens normal map as tangent-space RGB."""
        if self._singleLensNormal is None:
            self._GenerateSingleLensNormal()

        normalMap = self._singleLensNormal
        if hasattr(normalMap, "get"):
            normalMap = normalMap.get()

        import matplotlib.pyplot as plt

        image = ((normalMap + 1.0) * 0.5).clip(0.0, 1.0)
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title("Single Microlens Normal Map")
        plt.axis("off")
        plt.show()


