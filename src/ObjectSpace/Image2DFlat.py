

import PIL.Image
import OpenEXR, Imath

from .Points import PointsSource
from .Images import Image2D

from Util.Backend import backend as bd
from Util.Globals import (
    ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE,
    KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, ORIGIN, RNG
)
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor, RemoveBG
from Util.Misc import Magnitude, ArrayRotate, PolarToCartesian, RectPath
from Raytracing.RayBatch import RayBatch



class Image2DFlat(Image2D):
    def __init__(self):
        super().__init__()

        """Unsigned unit in mm. If anchors are not explicitly stated, assume image at infinity"""
        self.distance = INFINITY

        """Unsigned unit in degree. If anchors are not explicitly stated, assume image in 3D fills a horizontal angle of view. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40
        # Note that since this AoV describes the image and not the lens, decreasing this attribute will make the image smaller, as if the lens is having a higher AoV.

        """4 points data in Vec3. The 4 anchor points that pins the image in 3D space """
        self.pointAnchor = None

        self.imageCenter = None

        """Height/width of each pixel, assuming square pixels"""
        self.pixelPitch = None

        self._opacity = None
        self._opacityArray = None

    def LoadFrom8bit(self, imgPath):
        """
        For common 8 bit image formats like jpg, bmp, and png. If a png is not 8 bit, do not use this method. Find the right bit depth method instead.
        """

        # Read and save the original
        imgPath = RectPath(imgPath)
        self._fileMaster = PIL.Image.open(imgPath).convert("RGB")

        self._Update()

    def LoadFrom8BitPNG(self, imgPath):
        # Read and save the original
        imgPath = RectPath(imgPath)
        self._fileMaster = PIL.Image.open(imgPath).convert("RGBA")
        # if self._fileMaster.mode != "RGBA":
        #     self._fileMaster = self._fileMaster.convert("RGBA")

        r, g, b, self._opacity = self._fileMaster.split()

        self._fileMaster = self._fileMaster.convert("RGB")

        self._Update()

    def SetupTransitionTest(self, rotateDegree=45, scale=2):
        """
        This method adjusts the anchor points thus tilting the image. The tilted image can then be used to test transition.

        """

        # Create the 4 anchor points if they are not defined
        if (self.pointAnchor is None):
            self._CreateAnchors(self.distance)

        self.pointAnchor = ArrayRotate(bd.pi / 4,
                                       UP_DIR,
                                       self.imageCenter,
                                       self.pointAnchor)

        self._GeneratePointSources()

    def LoadFromEXR(self, imgPath):
        """
        Load only the RGB info from an EXR image. Other channels are ignored.
        """
        exrPath = RectPath(imgPath)

        exr = OpenEXR.InputFile(imgPath)

        # EXR header tells us the image size
        header = exr.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # EXR stores channels as strings like "R", "G", "B"
        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

        # Read raw channel strings
        r_str = exr.channel('R', FLOAT)
        g_str = exr.channel('G', FLOAT)
        b_str = exr.channel('B', FLOAT)

        # Convert to float32 NumPy arrays
        r = bd.frombuffer(r_str, dtype=bd.float32).reshape((height, width))
        g = bd.frombuffer(g_str, dtype=bd.float32).reshape((height, width))
        b = bd.frombuffer(b_str, dtype=bd.float32).reshape((height, width))

        # Stack to H×W×3
        rgb = bd.stack([r, g, b], axis=-1)
        self.rgbArray = bd.stack([r, g, b], axis=-1)

    def ReceiveAndEmitTowards(self, targets, incidents=None, sampleCount=64, useHighlightSources=False):
        """
        Receive an incident RayBatch, cull it against this flat image's opacity
        plane, then merge surviving incident rays with rays emitted by this image.
        """

        if useHighlightSources:
            emissionMethod = self.EmitHighlightSamplesTowards
        else:
            emissionMethod = self.EmitSamplesToward

        emitted = emissionMethod(targets, sampleCount)

        if (incidents is None) or (incidents.IsNoneType()):
            # When this is the furthest layer
            return emitted

        if self._opacityArray is None:
            return incidents.Copy().Merge(emitted)

        alpha_eps = 1e-6
        if not bd.any(self._opacityArray > alpha_eps):
            return incidents.Copy().Merge(emitted)

        if self.pointAnchor is None:
            self._CreateAnchors()

        p00 = self.pointAnchor[0]
        p10 = self.pointAnchor[1]
        p01 = self.pointAnchor[2]

        u_axis = p10 - p00
        v_axis = p01 - p00
        normal = bd.cross(u_axis, v_axis)

        positions = incidents.Position()
        directions = incidents.Direction()

        denom = directions @ normal
        valid_denom = bd.abs(denom) > 1e-8

        t_hit = ((p00 - positions) @ normal) / denom
        valid_hit = valid_denom & (t_hit > 0)

        if not bd.any(valid_hit):
            return incidents.Copy().Merge(emitted)

        hit_points = positions + t_hit[:, None] * directions
        rel = hit_points - p00

        u = (rel @ u_axis) / (u_axis @ u_axis)
        v = (rel @ v_axis) / (v_axis @ v_axis)

        inside = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
        valid_hit = valid_hit & inside

        if not bd.any(valid_hit):
            return incidents.Copy().Merge(emitted)

        H, W = self._opacityArray.shape
        x_img = u * (W - 1)
        y_img = v * (H - 1)

        x0 = bd.floor(x_img).astype(bd.int64)
        y0 = bd.floor(y_img).astype(bd.int64)

        x0 = bd.clip(x0, 0, W - 1)
        y0 = bd.clip(y0, 0, H - 1)

        x1 = bd.clip(x0 + 1, 0, W - 1)
        y1 = bd.clip(y0 + 1, 0, H - 1)

        fx = x_img - x0
        fy = y_img - y0

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        a00 = self._opacityArray[y0, x0]
        a10 = self._opacityArray[y0, x1]
        a01 = self._opacityArray[y1, x0]
        a11 = self._opacityArray[y1, x1]

        alpha_local = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11
        alpha_local = bd.clip(alpha_local, 0.0, 1.0)

        alpha_local = bd.where(valid_hit, alpha_local, bd.zeros_like(alpha_local))
        alpha_local = bd.where(alpha_local > alpha_eps, alpha_local, bd.zeros_like(alpha_local))

        rnd = RNG.rand(len(alpha_local))
        keep_mask = rnd >= alpha_local

        through = RayBatch(incidents.value[keep_mask])

        return through.Merge(emitted)

    # ==================================================================
    """ ====================== Private Methods ===================== """

    # ==================================================================

    def _Update(self):

        # Resize the input if needed
        if self.imageDimensionOverride is not None:
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            imageFile = self._fileMaster.resize((self.imageDimensionOverride, newHeight))
            if self._opacity is not None:
                if self.imageDimensionOverride is not None:
                    opacity_img = self._opacity.resize((self.imageDimensionOverride, newHeight))
                else:
                    opacity_img = self._opacity
                self._opacityArray = bd.array(opacity_img).astype(PRECISION_TYPE) / 255.0
            else:
                self._opacityArray = None
        else:
            self.imageDimensionOverride = self._fileMaster.width
            imageFile = self._fileMaster

        # Convert into array format
        self.rgbArray = bd.array(imageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in
        self.rgbArray = self.rgbArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        self._GeneratePointSources()

    def _GeneratePointSources(self):
        """
        Using the RGB and position data, generate a point source object that cooresponds to all the pixel/samples from the image input.
        """

        # Create the 4 anchor points if they are not defined
        if (self.pointAnchor is None):
            self._CreateAnchors()

        # This method of updating pixel pitch only works when the image is a spatial rectangle, is it stretches, then this will become uneven
        self.pixelPitch = Magnitude(self.pointAnchor[1] - self.pointAnchor[0]) / self.imageDimensionOverride

        sampleX = self.rgbArray.shape[1]
        sampleY = self.rgbArray.shape[0]

        u = bd.linspace(0, 1, sampleX)  # Interpolation values in x-direction
        v = bd.linspace(0, 1, sampleY)  # Interpolation values in y-direction

        # Create a meshgrid of interpolation factors
        U, V = bd.meshgrid(u, v, indexing="ij")  # Shape (sampleX, sampleY)

        # Compute the bilinear interpolation
        gridPositions = (
                (1 - U)[..., None] * (1 - V)[..., None] * self.pointAnchor[0].reshape(1, 1, 3) +
                U[..., None] * (1 - V)[..., None] * self.pointAnchor[1].reshape(1, 1, 3) +
                (1 - U)[..., None] * V[..., None] * self.pointAnchor[2].reshape(1, 1, 3) +
                U[..., None] * V[..., None] * self.pointAnchor[3].reshape(1, 1, 3)
        )

        # The grid generated this way is transposed, thus need the axis swapped
        gridPositions = bd.swapaxes(gridPositions, 0, 1)

        # Reshape the point position and color array
        gridPositions = gridPositions.reshape(sampleY * sampleX, 3)
        gridColors = self.rgbArray.reshape(sampleY * sampleX, 3)

        mask = None
        if self._opacityArray is not None:
            flat_opacity = self._opacityArray.reshape(sampleY * sampleX)
            mask = flat_opacity > 0
            gridPositions = gridPositions[mask]
            gridColors = gridColors[mask]

        aov_cols = []
        self.pointSourceAOVNames = []
        if self.AOVs is not None:
            for name in self.AOVNames:
                if name not in self.AOVs:
                    continue

                aov = self.AOVs[name]
                if aov.shape != (sampleY, sampleX):
                    raise ValueError(
                        f"Image2D._GeneratePointSources(): AOV '{name}' shape {aov.shape} "
                        f"does not match image shape {(sampleY, sampleX)}."
                    )

                flat_aov = aov.reshape(sampleY * sampleX)
                if mask is not None:
                    flat_aov = flat_aov[mask]

                aov_cols.append(flat_aov.reshape(-1, 1))
                self.pointSourceAOVNames.append(name)

        gridData = bd.concatenate([gridPositions, gridColors], axis=1)
        if aov_cols:
            gridData = bd.concatenate([gridData, bd.concatenate(aov_cols, axis=1)], axis=1)
        self.pointSource = PointsSource(gridData)

        # Concatenate the position and color
        # gridPositions = bd.concatenate([gridPositions, gridColors], axis=1)
        #
        # self.pointSource = PointsSource(gridPositions)

    def _CreateAnchors(self, zDist=None):

        # Infinty is not really workable, replace it with an approximation
        if (self.distance is INFINITY):
            zDist = -FAR_DISTANCE
        else:
            zDist = -bd.array(self.distance)

        rad = bd.deg2rad(self.horizontalAoV) / 2

        halfX = bd.abs(bd.tan(rad) * zDist)
        halfY = halfX * bd.abs(self._fileMaster.height / self._fileMaster.width)

        self.pointAnchor = bd.array([
            [-halfX, -halfY, zDist],
            [halfX, -halfY, zDist],
            [-halfX, halfY, zDist],
            [halfX, halfY, zDist],
        ])

        self.imageCenter = bd.mean(self.pointAnchor, axis=0)
