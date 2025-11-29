


import PIL.Image
import matplotlib.pyplot as plt
import OpenEXR, Imath

import sys
import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))

from .Points import PointsSource
from .Images import Image2D


from Util.Backend import backend as bd
from Util.Globals import ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE, KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, ORIGIN
from Util.PltPlot import DrawRaybatch, Setup3Dplot, AddXYZ, SetUnifScale, DrawPoints, DrawPointsPerColor, RemoveBG
from Util.Misc import Magnitude, ArrayRotate, PolarToCartesian, RectPath
from Raytracing.RayBatch import RayBatch








class Image2DVariDepth(Image2D):

    def __init__(self):
        """This extends from the flat image to contain varied depth."""
        super().__init__()

        """RGB array of the image"""
        self.rgbArray = None

        """This is the distance calculated from zArray and zDepthMappingRange."""
        self.zDistance = None

        """Z depth array of the image. This is the direct value of the image representing the z depth, not the acutal physical distance."""
        self.zArray = None

        """Alpha/opacity array of the image (optional, e.g. EXR)"""
        self.alphaArray = None

        """Because this is a secondary imaging process, an angle of view of the image source is needed. Value is unsigned unit in degree. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40

        """Flag: if True, zArray is already in physical units from EXR and should be used directly."""
        self._usingEXRDirectDepth = False

        """Master image file. For EXR this could include the alpha and the z depth"""
        self._fileMaster = None 

        """Separate file for the Z depth"""
        self._fileZ = None 

        """Point source object built from the image"""
        self.pointSource = None


        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


        """Z depth read form the input are in the range of [0, 1]. However, for actual imaging, this apparently is a not a valid distance range. This attribute is used to map the z depth into a more realistic range. By default, the range is set to 1.5m to 500m, i.e., typically portrait distance to infinity"""
        self.zDepthMappingRange = bd.array([KNOB_DISTANCE, FAR_DISTANCE])


        """For z-depth from rendered images, it typically means the distance from the object to the near clipping plane. To reconstruct the scene, it is thus needed to calculate the cone formed by the near clipping plane as well. 
        This property marks the distance from the near clipping plane to the camera, i.e.e, the (0, 0, 0) point. This should be an unsigned value."""
        self.nearClipping = 0


        """An array of same size with the number of point sources. Each entry in this array corresponds to the """
        self.jitterPerPoint = None


    def LoadFrom8bit(self, rgbImgPath, zImgPath=None):
        """
        For common 8 bit image formats like jpg, bmp, and png. Since these images tppically only contain the RGB information, the z depth information have to be read from a separate file. 

        :param rgbImgPath: Path to the RGB image file
        :param zImgPath: Path to the Z depth image file
        """

        self.LoadFrom8bitRGB(rgbImgPath)

        if(zImgPath is not None):
            self.LoadFrom8bitZ(zImgPath)

        self.Refresh()


    def LoadFrom8bitRGB(self, rgbImgPath):
        """
        Load an RGB image from the given path. 
        """
        rgbImgPath = RectPath(rgbImgPath)

        # Read and save the master file  
        self._fileMaster = PIL.Image.open(rgbImgPath).convert("RGB")
         # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            RGBImageFile = self._fileMaster.resize((self.imageDimensionOverride, newHeight))
        else:
            RGBImageFile = self._fileMaster

        # Convert into array format 
        self.rgbArray = bd.array(RGBImageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in 
        self.rgbArray = self.rgbArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        # 8-bit path: no alpha, and depth is NOT “direct EXR depth”
        self.alphaArray = None
        self._usingEXRDirectDepth = False


    def LoadFrom8bitZ(self, zImgPath):

        zImgPath = RectPath(zImgPath)

        # Read and save the z depth file 
        self._fileZ = PIL.Image.open(zImgPath).convert("L")

        # Resize the input if needed 
        if(self.imageDimensionOverride is not None):
            newHeight = int(self._fileMaster.height * (self.imageDimensionOverride / self._fileMaster.width))
            ZImageFile = self._fileZ.resize((self.imageDimensionOverride, newHeight))
        else:
            ZImageFile = self._fileZ

        # Convert into array format 
        self.zArray = bd.array(ZImageFile)

        # Normalize into [0, 1 range], this is where the 8 in 8 bit kicks in 
        self.zArray = self.zArray.astype(PRECISION_TYPE) / (TWO ** 8 - 1)

        self.UpdateDepthRange()


    def LoadFromEXR(self, exrPath, depthChannelNames=("Z", "depth", "Depth.Z", "depth.Z"),
                    alphaChannelNames=("A", "alpha", "Opacity")):

        exrPath = RectPath(exrPath)

        channelsFromEXR = self._ReadEXR(exrPath, depthChannelNames, alphaChannelNames)

        rgb = channelsFromEXR["rgb"]
        depth = channelsFromEXR["depth"]
        alpha = channelsFromEXR["alpha"]

        if rgb is None:
            raise ValueError(
                "EXR does not contain RGB channels ('R', 'G', 'B')."
            )
        if depth is None:
            raise ValueError(
                f"EXR does not contain any supported depth channels: {depthChannelNames}"
            )

        # ------------------------------------------------------------------
        # Optional resize to match imageDimensionOverride (nearest neighbor).
        # ------------------------------------------------------------------
        if self.imageDimensionOverride is not None:
            h, w, _ = rgb.shape
            new_w = int(self.imageDimensionOverride)
            new_h = int(h * (new_w / w))

            y_idx = bd.linspace(0, h - 1, new_h).astype(bd.int64)
            x_idx = bd.linspace(0, w - 1, new_w).astype(bd.int64)

            idx2d = bd.ix_(y_idx, x_idx)

            rgb = rgb[idx2d]  # (new_h, new_w, 3)
            depth = depth[idx2d]  # (new_h, new_w)
            if alpha is not None:
                alpha = alpha[idx2d]  # (new_h, new_w)

        # ------------------------------------------------------------------
        # Option B: use EXR depth directly (no [0,1] remapping).
        # zArray now stores the EXR depth values; zDistance is just the
        # signed version used by the ray system.
        # ------------------------------------------------------------------
        self.rgbArray = rgb.astype(PRECISION_TYPE)

        # EXR depth is assumed to be a positive distance (camera space or
        # near-plane space); system convention is negative toward object space.
        self.zArray = depth.astype(PRECISION_TYPE)
        self.zDistance = -self.zArray

        # Store alpha for later culling in _GeneratePolarPointSources
        if alpha is not None:
            self.alphaArray = alpha.astype(PRECISION_TYPE)
        else:
            self.alphaArray = None

        # Mark that this instance should *not* remap depth via zDepthMappingRange.
        self._usingEXRDirectDepth = True

        # Keep the original EXR path as "master" reference, this is due to
        # EXR not having a simple in-memory storage class like jpg/png.
        self._fileMaster = exrPath

        self.rgbArray = bd.flip(self.rgbArray, axis=(0,1))
        self.zArray = bd.flip(self.zArray, axis=(0,1))
        self.alphaArray = bd.flip(self.alphaArray, axis=(0,1))

        self.Refresh()


    def UpdateDepthRange(self, newRange=None):
        """
        Update the depth of the scene for non-EXR (8-bit) sources.
        For EXR Option B (direct depth), this method leaves zDistance alone.
        """

        # If we are in EXR direct-depth mode, do NOT touch zDistance.
        if getattr(self, "_usingEXRDirectDepth", False):
            # zDistance is already set from EXR depth; just ensure sign convention:
            self.zDistance = -self.zArray
            return

        # --- original behavior for normalized [0,1] zArray ---
        if(newRange is None):
            newRange = self.zDepthMappingRange

        if(newRange[0] > newRange[1]):
            self.zDepthMappingRange = bd.array([newRange[1], newRange[0]])
        else:
            self.zDepthMappingRange = bd.array(newRange)

        deltaRange = self.zDepthMappingRange[1] - self.zDepthMappingRange[0]

        self.zDistance = self.zArray * deltaRange + self.zDepthMappingRange[0]

        # Here the z Distance is unsigned, to make it work in the system, they need to be inverted. 
        self.zDistance = -self.zDistance


    def Refresh(self):
        """
        Manually refresh the parameters. Remap the depth (if needed) and recreate the point sources.
        """

        # Object space in world coordinate is negative, thus need to make
        # sure the near clip is also a negative number
        if (self.nearClipping > 0):
            self.nearClipping = - self.nearClipping

        # For EXR Option B, zDistance already comes from EXR depth, so skip
        # remapping; for 8-bit we still remap via zDepthMappingRange.
        if not getattr(self, "_usingEXRDirectDepth", False):
            self.UpdateDepthRange()
        else:
            # Ensure zDistance sign is consistent with current zArray
            self.zDistance = -self.zArray

        self._GeneratePolarPointSources()


    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.jitterPerPoint)


    def ReceiveAndEmitTowards(self, targets, incidents=None, sampleCount=64):
        """
        Receive an incident RayBatch, cull it and merge it with emitted RayBatch from this one.

        """

        if incidents is None:
            # When this is the furthest layer
            return self.EmitSamplesToward(targets, sampleCount)

        else:
            # Note: when this is called, it is safe to assume that this image has alpha info. It is also safe to assume that all incident rays are located further than this image, so no cross-intersection between the layers
            # Positions of incident rays (these are the source points on the
            # further-away image layer in object space).
            pos = incidents.Position()  # shape (N, 3)
            px = pos[:, 0]
            py = pos[:, 1]
            pz = pos[:, 2]

            # Origin of this secondary imaging system is the world origin.
            ox, oy, oz = ORIGIN  # (0, 0, 0)

            # Vector from origin to ray source (image pixel)
            vx = px - ox
            vy = py - oy
            vz = pz - oz

            # We want field angles relative to the -Z axis (object space at negative Z).
            # A simple way: flip the Z component when computing atan2 so that
            # "forward" is along -Z.
            eps = bd.array(1e-8, dtype=PRECISION_TYPE)
            vz_safe = bd.where(bd.abs(vz) < eps, eps, vz)

            theta_x = bd.arctan2(vx, -vz_safe)  # note the minus sign
            theta_y = bd.arctan2(vy, -vz_safe)  # note the minus sign

            # Same FOV convention as _GeneratePolarPointSources
            sampleY, sampleX, _ = self.rgbArray.shape  # (height, width, 3)

            horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
            half_horizontal = horizontalAoV_rad / 2.0

            verticalAoV_rad = horizontalAoV_rad * (sampleY / sampleX)
            half_vertical = verticalAoV_rad / 2.0

            # Rays whose angles fall outside the FOV cannot intersect this image
            inside_fov = (
                                 (theta_x >= -half_horizontal) & (theta_x <= half_horizontal)
                         ) & (
                                 (theta_y >= -half_vertical) & (theta_y <= half_vertical)
                         )

            # Map angles in [-half, half] → normalized [0, 1]
            u = (theta_x + half_horizontal) / (horizontalAoV_rad + eps)
            v = (theta_y + half_vertical) / (verticalAoV_rad + eps)

            # Convert to pixel indices
            x = (u * sampleX).astype(bd.int64)
            y = (v * sampleY).astype(bd.int64)

            # Clamp to valid range
            x = bd.clip(x, 0, sampleX - 1)
            y = bd.clip(y, 0, sampleY - 1)

            # Look up alpha per incident ray
            alpha_flat = bd.flip(self.alphaArray, axis=(0, 1))[y, x]

            # Blocked if inside FOV and pixel is opaque (alpha > 0)
            blocked = inside_fov & (alpha_flat > 0)
            keep_mask = ~blocked

            # Keep only rays that are not blocked by this layer
            through = incidents.Copy()
            through.Mask(keep_mask)

            # ------------------------------------------------------------------
            # 2. Emit new rays from this image toward the targets
            # ------------------------------------------------------------------
            emitted = self.EmitSamplesToward(targets, sampleCount)

            # ------------------------------------------------------------------
            # 3. Return union: surviving incident rays + newly emitted rays
            # ------------------------------------------------------------------
            return through.Merge(emitted)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ReadEXR(self, exrPath, depthChannelNames=("Z", "depth", "Depth.Z", "depth.Z"),
                    alphaChannelNames=("A", "alpha", "Opacity")):
        """
        Read an EXR image and its channels.
        """
        # self.debug_show_exr_channels(exrPath)

        exr = OpenEXR.InputFile(exrPath)
        header = exr.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
        available = list(header["channels"].keys())

        def read_channel(name):
            """Return H×W float32 array, or None if missing."""
            if name not in available:
                return None
            arr = bd.frombuffer(exr.channel(name, FLOAT), dtype=bd.float32)
            return arr.reshape(height, width)

        # -------------------------------- Read RGB --------------------------------
        r = read_channel("R")
        g = read_channel("G")
        b = read_channel("B")

        if r is not None and g is not None and b is not None:
            rgb = bd.stack([r, g, b], axis=-1)  # (H, W, 3)
        else:
            rgb = None

        # ----------- Read Depth (supports multiple naming conventions) -----------
        depth = None
        for dname in depthChannelNames:
            d = read_channel(dname)
            if d is not None:
                depth = d
                break

        # ----------------------------Read Alpha/Opacity ---------------------------
        alpha = None
        for aname in alphaChannelNames:
            a = read_channel(aname)
            if a is not None:
                alpha = a
                break

        return {
            "rgb": rgb,
            "r": r,
            "g": g,
            "b": b,
            "alpha": alpha,
            "depth": depth,
            "channels": available,
        }


    def _GeneratePolarPointSources(self):
        """
        Generate point sources from the image where each pixel is represented 
        in polar coordinates: [theta_x, theta_y, D, R, G, B]. 
        Here, theta_x and theta_y are field angles (in radians) relative to the optical axis,
        and D is the distance from the front vertex.
        """
        sampleY, sampleX, _ = self.rgbArray.shape  # (height, width, channels)

        # Field of view in radians
        horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
        half_horizontal = horizontalAoV_rad / 2.0

        verticalAoV_rad = horizontalAoV_rad * (sampleY / sampleX)
        half_vertical = verticalAoV_rad / 2.0

        # Pixel-center coordinates in [0,1]
        x_idx = bd.arange(sampleX, dtype=PRECISION_TYPE)
        y_idx = bd.arange(sampleY, dtype=PRECISION_TYPE)

        u = (x_idx + 0.5) / sampleX  # shape (sampleX,)
        v = (y_idx + 0.5) / sampleY  # shape (sampleY,)

        # Grid in (y, x) order so axis 0 = rows, axis 1 = columns
        U, V = bd.meshgrid(u, v, indexing="xy")  # both (sampleY, sampleX)

        # Angles per pixel
        theta_x = -half_horizontal + U * horizontalAoV_rad  # (sampleY, sampleX)
        theta_y = -half_vertical + V * verticalAoV_rad  # (sampleY, sampleX)

        # Distance: no swapaxes needed; keep same (sampleY, sampleX) layout
        zClipDist = self._zClipDistance(
            half_horizontal, half_vertical, sampleY, sampleX, self.nearClipping
        )  # (sampleY, sampleX)

        D = self.zDistance + bd.swapaxes(zClipDist, 0, 1)  # (sampleY, sampleX)

        # Jitter in same layout, then flatten
        self.jitterPerPoint = self._AngularJitter(
            half_horizontal, half_vertical, sampleY, sampleX, D
        ).reshape(sampleY * sampleX)

        # Stack [theta_x, theta_y, D] and flatten
        coordinates = bd.stack([theta_x, theta_y, D], axis=-1)  # (sampleY, sampleX, 3)
        coordinates = coordinates.reshape(sampleY * sampleX, 3)

        # Colors flattened in same order
        colors = self.rgbArray.reshape(sampleY * sampleX, 3)

        if self.alphaArray is not None:
            alpha_flat = self.alphaArray.reshape(sampleY * sampleX)
            keep_mask = alpha_flat > 0

            coordinates = coordinates[keep_mask]
            colors = colors[keep_mask]
            self.jitterPerPoint = self.jitterPerPoint[keep_mask]

        points = bd.concatenate([coordinates, colors], axis=1)

        self.pointSource = PointsSource(points)
        self.pointSource.isCartesian = False
        self.pointSource.angleInRad = True


    def _zClipDistance(self, horizontalHalfRad, verticalHalfRad,
                       h_steps, v_steps, clipDepth):
        """
        Calculate the distance from the camera pivot to the source points' projection on the near clipping plane.
        The rectangle sits in the XZ-plane at y = 0.
        Its corners subtend ±(horizontal_FOV/2) in X and ±(vertical_FOV/2) in Z
        from the origin *when projected onto a plane located at z = plane_z*.

        :param horizontalHalfRad: Half horizontal field-angle (rad).
        :param verticalHalfRad: Half vertical field-angle (rad).
        :param h_steps: Sub-divisions along the horizontal direction (number of intervals).
        :param v_steps: Sub-divisions along the vertical   direction (number of intervals).
        :param clipDepth: The Z-coordinate (“depth”) where the rectangle plane is located. All grid points share this z.

        :return distances: Euclidean distance from P to each rectangle node.
        """

        if(clipDepth >= 0): clipDepth = -clipDepth

        theta_h = bd.linspace(-horizontalHalfRad, horizontalHalfRad, h_steps )  # shape (h_steps,)
        theta_v = bd.linspace(-verticalHalfRad, verticalHalfRad, v_steps )  # shape (v_steps,)

        # 2. Meshgrid (X uses theta_h, Z uses theta_v)
        TH, TV = bd.meshgrid(theta_h, theta_v, indexing='xy')  # TH,TV shape (v, h)

        # 3. Convert angular deflection to X and Z on plane_z
        #    x = (plane_z) * tan(theta_h)
        #    z = plane_z + (plane_z) * tan(theta_v)
        #    (y is zero because rectangle lies in XZ plane)
        X = clipDepth * bd.tan(TH)  # shape (v, h)
        Y = clipDepth * (1 + bd.tan(TV))  # add base plane depth
        Z = bd.zeros_like(X)  # same shape

        # 4. Euclidean distance to P = (0,0,z0)
        distances = bd.sqrt(X**2 + Y**2 + clipDepth**2)  # y-term is zero

        return distances


    def _AngularJitter(self, horizontalHalfRad, verticalHalfRad,
                       h_steps, v_steps, perPointDistance):
        """
        Calculating jitter based on their angular resolution and distance.
        """

        # angular pitch of a single pixel
        #  (use max(n-1,1) to avoid division by zero on 1-pixel edges)
        dtheta_h = (horizontalHalfRad * 2.0) / max(h_steps - 1, 1)
        # dtheta_v = (verticalHalfRad * 2.0) / max(v_steps - 1, 1)

        # ± jitter range that keeps a ray inside its pixel’s solid-angle
        jitter_h = perPointDistance * bd.tan(dtheta_h * 0.5)  # shape (v, h)
        # jitter_v = perPointDistance * bd.tan(dtheta_v * 0.5)  # shape (v, h)

        # flatten to 1-D so index order matches the flattened point list
        jitter_h_flat = jitter_h.reshape(-1)
        # jitter_v_flat = bd.swapaxes(jitter_v, 0, 1).reshape(-1)

        # Horizontal and vertical angular resolution should be the same, a single return should suffice in most cases.
        return jitter_h_flat


    def debug_show_exr_channels(self, exr_path):
        """
        Prints all channels in the EXR and displays each as an image.
        Handles float and half precision.
        """

        print(f"\n--- Reading EXR: {exr_path} ---")

        exr = OpenEXR.InputFile(exr_path)
        header = exr.header()

        channels = list(header["channels"].keys())
        print("Found channels:")
        for c in channels:
            print(f"  • {c}")

        # get image size
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1

        # Display each channel
        n = len(channels)
        cols = 4
        rows = int(bd.ceil(n / cols))

        plt.figure(figsize=(4 * cols, 4 * rows))

        for i, chan in enumerate(channels, start=1):

            # Auto-detect pixel type
            ch = header["channels"][chan]
            if ch.type.v == Imath.PixelType(Imath.PixelType.FLOAT).v:
                pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
                dtype = bd.float32
            else:
                pixel_type = Imath.PixelType(Imath.PixelType.HALF)
                dtype = bd.float16

            # Read channel
            arr = bd.frombuffer(exr.channel(chan, pixel_type), dtype=dtype)
            arr = arr.reshape(height, width)

            # Normalize for display
            arr_disp = arr.astype(bd.float32)
            if bd.any(arr_disp > 0):
                arr_disp = arr_disp / bd.max(arr_disp)

            ax = plt.subplot(rows, cols, i)
            ax.imshow(arr_disp, cmap="viridis")
            ax.set_title(chan)
            ax.axis("off")

        plt.tight_layout()
        plt.show()


def main():
    
    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    img = Image2DVariDepth()
    img.imageDimensionOverride = 200 
    #img.zDepthMappingRange = bd.array([500, 1000])

    img.LoadFromEXR("allChannels.exr")

    img.DrawImage()

    RemoveBG()
    SetUnifScale(1000)
    plt.show()




if __name__ == "__main__":
    main()