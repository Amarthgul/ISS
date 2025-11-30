


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

        """Z depth array of the image. 
        This is the direct value of the image representing the z depth, not the actual physical distance used to calculate. Although in the case of EXR, zArray and zDistance is regarded as the same."""
        self.zArray = None

        """Alpha/opacity array of the image (optional, e.g. EXR)"""
        self.alphaArray = None

        """Other channels when reading from an EXR file."""
        self.AOVs = None

        """Because this is a secondary imaging process, an angle of view of the image source is needed. Value is unsigned unit in degree. Default value 40 degrees, which is a 50mm on 135 format."""
        self.horizontalAoV = 40

        """Stores the calculated 3D Cartesian coordinates (X, Y, Z) of the opaque pixels."""
        self.geometry3d = None

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


    def LoadFromEXR(self, exrPath, depthChannelNames=("Z", "Z.R", "depth", "Depth.Z", "depth.Z"),
                    alphaChannelNames=("A", "alpha", "Opacity")):

        exrPath = RectPath(exrPath)

        channelsFromEXR = self._ReadEXR(exrPath, depthChannelNames, alphaChannelNames)

        rgb = channelsFromEXR["rgb"]
        depth = channelsFromEXR["depth"]
        alpha = channelsFromEXR["alpha"]
        aov_dict = channelsFromEXR.get("AOVs", {})

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

            rgb = rgb[idx2d]    # (new_h, new_w, 3)
            depth = depth[idx2d]  # (new_h, new_w)
            if alpha is not None:
                alpha = alpha[idx2d]  # (new_h, new_w)

            # Resize all additional AOV channels the same way
            resized_aovs = {}
            for name, arr in aov_dict.items():
                resized_aovs[name] = arr[idx2d]
            aov_dict = resized_aovs

        # ------------------------------------------------------------------
        # EXR depth is assumed to be in physical units already.
        # zArray stores EXR depth; zDistance is signed (negative towards object).
        # ------------------------------------------------------------------
        self.rgbArray = rgb.astype(PRECISION_TYPE)

        self.zArray = depth.astype(PRECISION_TYPE)
        self.zDistance = -self.zArray

        # Alpha for opacity
        if alpha is not None:
            self.alphaArray = alpha.astype(PRECISION_TYPE)
        else:
            self.alphaArray = None

        # Additional AOVs (all channels except RGB, depth, alpha)
        if aov_dict:
            self.AOVs = {name: arr.astype(PRECISION_TYPE) for name, arr in aov_dict.items()}
        else:
            self.AOVs = None

        # Mark direct-depth EXR mode
        self._usingEXRDirectDepth = True

        # Keep the original EXR path as "master" reference
        self._fileMaster = exrPath

        # Flip vertically to match system indexing (y,x)
        self.rgbArray = bd.flip(self.rgbArray, axis=(0, 1))
        self.zArray = bd.flip(self.zArray, axis=(0, 1))
        if self.alphaArray is not None:
            self.alphaArray = bd.flip(self.alphaArray, axis=(0, 1))
        if self.AOVs is not None:
            for name in list(self.AOVs.keys()):
                self.AOVs[name] = bd.flip(self.AOVs[name], axis=(0, 1))

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


    def ReceiveAndEmitTowards(self, targets, incidents:RayBatch=None, sampleCount:int=64):
        """
        Receive an incident RayBatch, cull it and merge it with emitted RayBatch from this one.

        """

        if (incidents is None) or (incidents.IsNoneType()):
            # When this is the furthest layer
            return self.EmitSamplesToward(targets, sampleCount)

        else:
            pos = incidents.Position()  # shape (N, 3)
            px = pos[:, 0]
            py = pos[:, 1]
            pz = pos[:, 2]  # The Z-position of the incident ray source (farther layer)

            # Origin of this secondary imaging system is the world origin (0, 0, 0)
            ox, oy, oz = ORIGIN

            # Vector from origin to ray source (image pixel)
            vx = px - ox
            vy = py - oy
            vz = pz - oz

            # We want field angles relative to the -Z axis (object space at negative Z).
            eps = bd.array(1e-8, dtype=PRECISION_TYPE)
            vz_safe = bd.where(bd.abs(vz) < eps, eps, vz)

            theta_x = bd.arctan2(vx, -vz_safe)  # note the minus sign
            theta_y = bd.arctan2(vy, -vz_safe)  # note the minus sign

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
            x_center = (u * sampleX).astype(bd.int64)
            y_center = (v * sampleY).astype(bd.int64)

            # --- START: Conservative 3x3 Check ---

            # Initialize a mask for blocked rays (Assume NOT blocked initially)
            blocked = bd.zeros_like(inside_fov, dtype=bd.bool_)

            # NOTE: We use the system's Z-convention: negative Z for object space.
            # Incident ray position Pz is the ray source, on a farther layer.
            # Stored Z-distance is self.zDistance (negative Z values).

            # Loop over 3x3 neighborhood (dx, dy from -1 to 1)
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    # Calculate neighbor pixel indices
                    x_neighbor = bd.clip(x_center + dx, 0, sampleX - 1)
                    y_neighbor = bd.clip(y_center + dy, 0, sampleY - 1)

                    # 1. Sample Z-Depth at neighbor (self.zDistance is already negative)
                    # We flip the array axes (0 and 1) to match the (y, x) indexing used for sampling
                    z_dist_array = bd.flip(self.zDistance, axis=(0, 1))
                    z_stored = z_dist_array[y_neighbor, x_neighbor]  # This is the stored Z (negative value)

                    # 2. Sample Alpha at neighbor
                    alpha_array = bd.flip(self.alphaArray, axis=(0, 1))
                    alpha_stored = alpha_array[y_neighbor, x_neighbor]

                    # Culling Condition Check:
                    # Ray is blocked if:
                    # a) It is inside the FOV
                    # b) The ray's source Z (pz) is BEHIND (more negative) the stored Z (z_stored)
                    # c) The pixel is opaque (alpha_stored > 0)

                    # Since both pz and z_stored are negative:
                    # pz < z_stored  (e.g., pz = -20, z_stored = -10, then -20 < -10 is TRUE)
                    # This means the ray source is closer to the image, which is WRONG for culling.

                    # Correct Culling Condition (Ray source is FARTHER than the stored depth):
                    # pz > z_stored  (e.g., pz = -10, z_stored = -20, then -10 > -20 is TRUE)
                    # pz is the current ray position (source) on a FARTHER layer.
                    # z_stored is the surface of THIS layer.
                    # If pz > z_stored, the ray passes *through* the stored surface.

                    hit_and_opaque = (
                            (
                                        pz < z_stored) &  # The ray source Z must be MORE negative (farther) than the stored surface Z
                            (alpha_stored > 0)
                    )

                    # Conservatively update the blocked mask: if *any* neighbor blocks it, the ray is blocked.
                    blocked = blocked | (inside_fov & hit_and_opaque)

            # --- END: Conservative 3x3 Check ---

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


    def GetAOVNames(self):
        """
        Return all EXR channel names except the basic RGB ('R','G','B').
        Used later when saving the rendered image (including depth, alpha, and other AOVs).
        """
        if self._fileMaster is None:
            return []

        try:
            exr = OpenEXR.InputFile(self._fileMaster)
        except Exception:
            # Not an EXR master, or cannot be opened
            return []

        header = exr.header()
        channels = list(header["channels"].keys())

        # Exclude only the basic RGB channels
        aov_names = [c for c in channels if c not in ("R", "G", "B")]
        return aov_names



    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ReadEXR(self, exrPath, depthChannelNames, alphaChannelNames):
        """
        Read an EXR image and its channels.
        """
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
        depth_name_sel = None
        for dname in depthChannelNames:
            d = read_channel(dname)
            if d is not None:
                depth = d
                depth_name_sel = dname
                break

        # ----------------------------Read Alpha/Opacity ---------------------------
        alpha = None
        alpha_name_sel = None
        for aname in alphaChannelNames:
            a = read_channel(aname)
            if a is not None:
                alpha = a
                alpha_name_sel = aname
                break

        # ------------------------ Read additional AOV channels --------------------
        used_names = set()
        if r is not None and g is not None and b is not None:
            used_names.update(["R", "G", "B"])
        if depth is not None and depth_name_sel is not None:
            used_names.add(depth_name_sel)
        if alpha is not None and alpha_name_sel is not None:
            used_names.add(alpha_name_sel)

        aov_dict = {}
        for ch_name in available:
            if ch_name in used_names:
                continue
            arr = read_channel(ch_name)
            if arr is not None:
                aov_dict[ch_name] = arr  # H×W

        return {
            "rgb": rgb,
            "r": r,
            "g": g,
            "b": b,
            "alpha": alpha,
            "depth": depth,
            "channels": available,
            "AOVs": aov_dict,
            "depth_name": depth_name_sel,
            "alpha_name": alpha_name_sel,
        }


    def _GeneratePolarPointSources(self):
        """
        Generate point sources from the image where each pixel is represented 
        in polar coordinates: [theta_x, theta_y, D, R, G, B, alpha, depth, AOV...].
        theta_x and theta_y are field angles (in radians) relative to the optical axis,
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

        # Jitter in same layout, then flatten (pixel-wise)
        self.jitterPerPoint = self._AngularJitter(
            half_horizontal, half_vertical, sampleY, sampleX, D
        ).reshape(sampleY * sampleX)

        # Stack [theta_x, theta_y, D] and flatten
        coordinates = bd.stack([theta_x, theta_y, D], axis=-1)  # (sampleY, sampleX, 3)
        coordinates = coordinates.reshape(sampleY * sampleX, 3)

        # Colors flattened in same order
        colors = self.rgbArray.reshape(sampleY * sampleX, 3)

        # Prepare alpha, depth, and extra AOVs as flattened arrays
        N = sampleY * sampleX

        alpha_flat = None
        if self.alphaArray is not None:
            alpha_flat = self.alphaArray.reshape(N)

        depth_flat = self.zArray.reshape(N)  # EXR depth (or mapped depth for 8-bit)

        extra_aov_flats = []
        if self.AOVs is not None:
            for name, arr in self.AOVs.items():
                extra_aov_flats.append(arr.reshape(N))

        # If alpha is present, cull transparent pixels and apply the same mask to all arrays
        if alpha_flat is not None:
            keep_mask = alpha_flat > 0

            coordinates = coordinates[keep_mask]
            colors = colors[keep_mask]
            self.jitterPerPoint = self.jitterPerPoint[keep_mask]

            alpha_flat = alpha_flat[keep_mask]
            depth_flat = depth_flat[keep_mask]
            for i in range(len(extra_aov_flats)):
                extra_aov_flats[i] = extra_aov_flats[i][keep_mask]
        else:
            keep_mask = None  # not used, but kept for clarity

        # Build AOV columns: alpha, depth, then all other AOVs
        aov_cols = []

        # alpha as first AOV column (if available)
        if alpha_flat is not None:
            aov_cols.append(alpha_flat.reshape(-1, 1))

        # depth as second AOV column
        aov_cols.append(depth_flat.reshape(-1, 1))

        # other AOVs (in the order stored in self.AOVs)
        for flat in extra_aov_flats:
            aov_cols.append(flat.reshape(-1, 1))

        if aov_cols:
            aov_matrix = bd.concatenate(aov_cols, axis=1)  # (num_points, num_aovs)
            points = bd.concatenate([coordinates, colors, aov_matrix], axis=1)
        else:
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


    def _PrintChannels(self, exr_path):
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