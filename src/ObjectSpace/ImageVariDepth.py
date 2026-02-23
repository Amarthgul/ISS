


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
from Util.Globals import (
    ZERO, ONE, TWO, INIT_ELLIPSE_TILT, INFINITY, FAR_DISTANCE,
    KNOB_DISTANCE, PRECISION_TYPE, UP_DIR, Axis, ORIGIN, RNG
)
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

        """When using EXR, the unit in Z may not be the same."""
        self.zUnitConversion = 10

        """When set to an int, the image object will be resampled with image width replaced with this attribute"""
        self.imageDimensionOverride = None 


        """Z depth read form the input are in the range of [0, 1]. However, for actual imaging, this apparently is a not a valid distance range. This attribute is used to map the z depth into a more realistic range. By default, the range is set to 1.5m to 500m, i.e., typically portrait distance to infinity"""
        self.zDepthMappingRange = bd.array([KNOB_DISTANCE, FAR_DISTANCE])


        """For z-depth from rendered images, it typically means the distance from the object to the near clipping plane. To reconstruct the scene, it is thus needed to calculate the cone formed by the near clipping plane as well. 
        This property marks the distance from the near clipping plane to the camera, i.e.e, the (0, 0, 0) point. This should be an unsigned value."""
        self.nearClipping = 0


        """Certain z depth are not possible but might be present due to numerical errors. This attribute help clips them by mandating a limit for the first distance in the image. """
        self.zFarLimit = 1e6


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

        self.zArray = depth.astype(PRECISION_TYPE) * self.zUnitConversion
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
        # self.rgbArray = bd.flip(self.rgbArray, axis=0)
        # self.zDistance = bd.flip(self.zDistance, axis=(0, 1))
        #self.zArray = bd.flip(self.zArray, axis=(0, 1))
        self.alphaArray = bd.flip(self.alphaArray, axis=(0, 1))

        # if self.AOVs is not None:
        #     for name in list(self.AOVs.keys()):
        #         self.AOVs[name] = bd.flip(self.AOVs[name], axis=(0, 1))

        self._GeneratePolarPointSources()



    def EmitSamplesToward(self, targets, sampleCount=64):

        return self.pointSource.EmitSamplesToward(targets, sampleCount, self.jitterPerPoint)


    def ReceiveAndEmitTowards(self, targets, incidents:RayBatch=None, sampleCount:int=64):
        """
        Receive an incident RayBatch, cull it and merge it with emitted RayBatch from this one.

        """

        emitted = self.EmitSamplesToward(targets, sampleCount)

        if (incidents is None) or (incidents.IsNoneType()):
            # When this is the furthest layer
            return emitted

        else:
            through = self._CullSelfOcclusionVariDepth(incidents)

            emitted = self.EmitSamplesToward(targets, sampleCount)

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


    def FloodDepth(self, value):
        """
        Flood-fill the depth buffer zArray and its derived zDistance
        with a constant depth 'value'.

        Useful for debugging mis-occlusion or alpha vs depth alignment.
        """
        if self.zArray is None:
            return

        # Fill zArray (the depth values in world Z units)
        self.zArray[:] = bd.asarray(value, dtype=PRECISION_TYPE)

        # Update zDistance = -zArray (your convention)
        self.zDistance = -self.zArray

        self._GeneratePolarPointSources()


    def DrawMask(self, alpha_eps=1e-6, show_zdistance=True, use_log=False, max_markers=2000):
        """
        Debug helper for the exact z_valid / z_min / z_max logic used in _CullSelfOcclusionVariDepth().

        It reconstructs:
            alpha_valid = bd.flip(self.alphaArray, axis=(0,1)) > alpha_eps   (same as cull)
            z_valid     = self.zDistance[alpha_valid]
            z_min/z_max = min/max(z_valid)

        Then displays:
          1) alphaArray (as stored)
          2) alpha_valid used by the culler (after that extra flip)
          3) depth (zDistance or zArray) masked by alpha_valid (NaN outside)
          4) an outlier/min/max locator map (where masked depth equals min/max; plus optional outliers)

        Params:
          alpha_eps: threshold for alpha_valid
          show_zdistance: if True visualize zDistance; else visualize zArray
          use_log: if True, visualize log10(abs(depth)) for readability
          max_markers: cap the number of highlighted pixels to keep plotting responsive
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self.alphaArray is None:
            print("[_DrawMask] alphaArray is None")
            return
        if self.zDistance is None or self.zArray is None:
            print("[_DrawMask] zDistance/zArray is None")
            return

        # --- backend->numpy conversion (cupy compatible)
        def to_np(a):
            try:
                return a.get()
            except Exception:
                try:
                    return bd.asnumpy(a)
                except Exception:
                    return np.asarray(a)

        # Pick which depth channel to inspect
        depth = self.zDistance if show_zdistance else self.zArray

        # IMPORTANT: replicate the culler exactly
        # _CullSelfOcclusionVariDepth currently does this: :contentReference[oaicite:2]{index=2}
        # alpha_valid = bd.flip(self.alphaArray, axis=(0, 1)) > alpha_eps
        alpha_valid = (bd.flip(self.alphaArray, axis=(0, 1)) > alpha_eps) & (bd.abs(self.zDistance) < self.zFarLimit*self.zUnitConversion)

        # Create masked depth map in the SAME index space as depth (i.e., using alpha_valid directly)
        depth_masked = bd.where(alpha_valid, depth, bd.nan)

        # Extract z_valid exactly like the culler
        z_valid = depth[alpha_valid]
        if not bd.any(alpha_valid):
            print("[_DrawMask] alpha_valid empty -> z_valid empty")
            return

        z_min = bd.min(z_valid)
        z_max = bd.max(z_valid)

        # Convert to numpy for plotting
        alpha_np = to_np(self.alphaArray)
        alpha_valid_np = to_np(alpha_valid).astype(np.float32)
        depth_np = to_np(depth)
        depth_masked_np = to_np(depth_masked)

        # Helper for visualization scaling
        def depth_vis(arr):
            arr = arr.copy()
            fin = np.isfinite(arr)
            if use_log:
                # log10 of absolute magnitude, preserve sign by reapplying sign later if needed
                sgn = np.sign(arr[fin])
                arr[fin] = np.log10(np.maximum(np.abs(arr[fin]), 1e-12)) * sgn
            return arr

        depth_vis_np = depth_vis(depth_np)
        depth_masked_vis_np = depth_vis(depth_masked_np)

        # Robust color limits
        def robust_limits(arr):
            fin = np.isfinite(arr)
            if not fin.any():
                return 0.0, 1.0
            vmin = np.percentile(arr[fin], 1.0)
            vmax = np.percentile(arr[fin], 99.0)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = float(np.min(arr[fin]))
                vmax = float(np.max(arr[fin]))
            return vmin, vmax

        vmin_raw, vmax_raw = robust_limits(depth_vis_np)
        vmin_m, vmax_m = robust_limits(depth_masked_vis_np)

        # Min/max location map inside alpha_valid
        zmin_f = float(to_np(z_min))
        zmax_f = float(to_np(z_max))

        # Build locator maps: where masked depth equals min/max (exact compare can be noisy on floats;
        # use a tolerance based on range)
        masked_fin = np.isfinite(depth_masked_np)
        if masked_fin.any():
            rng = np.nanmax(depth_masked_np) - np.nanmin(depth_masked_np)
            tol = max(1e-6, float(rng) * 1e-6)
        else:
            tol = 1e-6

        is_min = masked_fin & (np.abs(depth_masked_np - zmin_f) <= tol)
        is_max = masked_fin & (np.abs(depth_masked_np - zmax_f) <= tol)

        locator = np.zeros_like(alpha_valid_np, dtype=np.float32)
        locator[is_min] = 1.0  # min pixels
        locator[is_max] = 2.0  # max pixels

        # Cap marker density (optional)
        if locator.sum() > max_markers:
            # randomly subsample markers for visibility
            ys, xs = np.where(locator > 0)
            idx = np.random.choice(len(xs), size=max_markers, replace=False)
            locator2 = np.zeros_like(locator)
            locator2[ys[idx], xs[idx]] = locator[ys[idx], xs[idx]]
            locator = locator2

        # Plot
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        plt.title("alphaArray (stored)")
        plt.imshow(alpha_np)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 2)
        plt.title("alpha_valid used by _CullSelfOcclusionVariDepth (flip+threshold)")
        plt.imshow(alpha_valid_np)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 3)
        chan = "zDistance" if show_zdistance else "zArray"
        plt.title(f"{chan} masked by alpha_valid ({'log' if use_log else 'linear'})")
        plt.imshow(depth_masked_vis_np, vmin=vmin_m, vmax=vmax_m)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, 2, 4)
        plt.title("Min/Max locator inside z_valid (1=min, 2=max)")
        plt.imshow(locator)
        plt.colorbar(fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

        # Print the exact min/max the culler would use
        print(f"[_DrawMask] z_valid count = {int(np.sum(alpha_valid_np > 0))}")
        print(f"[_DrawMask] z_min = {zmin_f}, z_max = {zmax_f}, tol={tol}")


    def Stats(self):
        return self.pointSource.Stats()


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================

    def _CullSelfOcclusionVariDepth(self, incidents: RayBatch, iterStep: int = 3):
        """
        Cull self-occluded rays against this vari-depth image using a
        volumetric alpha process, but fully vectorized over rays.

        For each incident ray:
          1. Find the segment [t0,t1] where the ray overlaps the image's z-range.
          2. March along that segment in 'iterStep' samples (all rays in parallel).
          3. For each sample, map to the image using AoV and bilinearly sample alpha.
          4. Accumulate transmittance per ray: T = Π (1 - alpha_step).
          5. Let alpha_acc = 1 - T in [0,1]. Interpret alpha_acc as the probability
             to DROP the ray (self-occluded).
          6. Monte Carlo: draw U ~ Uniform[0,1], drop if U < alpha_acc.

        Returns a new RayBatch containing only the rays that survive.
        """

        # If no depth or alpha, nothing to cull.
        if (self.zDistance is None) or (self.alphaArray is None):
            return incidents.Copy()

        alpha_eps = 1e-6

        # Pixels that actually have some opacity.
        # alpha_valid = self.alphaArray > alpha_eps
        alpha_valid = (bd.flip(self.alphaArray, axis=(0, 1)) > alpha_eps) & (bd.abs(self.zDistance) < self.zFarLimit*self.zUnitConversion)
        if not bd.any(alpha_valid):
            # Entire image effectively transparent
            return incidents.Copy()

        # Define z-slab using only alpha>0 pixels (ignores 99990-style sentinels from EXR).
        z_valid = self.zDistance[alpha_valid]
        z_min = bd.min(z_valid)
        z_max = bd.max(z_valid)
        # print("Depth min max at ", z_min, "  ", z_max)

        if float(z_min) >= float(z_max):
            # Degenerate z-range, nothing to intersect.
            return incidents.Copy()

        # ------------------------------------------------------------------
        # Gather ray data
        # ------------------------------------------------------------------
        positions = incidents.Position()  # (N,3)
        directions = incidents.Direction()  # (N,3)
        N = positions.shape[0]

        oz = positions[:, 2]
        dz = directions[:, 2]

        # Rays nearly parallel to the z-slab can't intersect it reliably.
        dz_eps = 1e-8
        valid_dz = bd.abs(dz) > dz_eps

        # Parametric intersection with z-slab [z_min,z_max] for all rays:
        # r(t) = o + t d,  z(o + t d) ∈ [z_min,z_max]
        t_enter = (z_min - oz) / dz
        t_exit = (z_max - oz) / dz

        t0 = bd.minimum(t_enter, t_exit)
        t1 = bd.maximum(t_enter, t_exit)

        # Only care about forward segments (t>0), and we need t1>t0
        has_seg = valid_dz & (t1 > 0)
        # Clamp t0 to 0 for forward marching
        t0 = bd.where(t0 < 0, bd.zeros_like(t0), t0)
        has_seg = has_seg & (t1 > t0)

        # If no ray overlaps the slab, just return.
        if not bd.any(has_seg):
            return incidents.Copy()

        # ------------------------------------------------------------------
        # Prepare AoV mapping constants
        # ------------------------------------------------------------------
        H, W = self.alphaArray.shape

        horizontalAoV_rad = bd.deg2rad(self.horizontalAoV)
        half_horizontal = horizontalAoV_rad / 2.0

        # Vertical FoV derived from aspect ratio
        verticalAoV_rad = horizontalAoV_rad * (H / W)
        half_vertical = verticalAoV_rad / 2.0

        # ------------------------------------------------------------------
        # Vectorized marching: build (N,steps) sample parameters t_mid
        # ------------------------------------------------------------------
        steps = max(int(iterStep), 1)

        length = t1 - t0  # (N,)
        # For rays without segment, length may be garbage; mask them later.
        # Normalized step centers in [0,1], shape (1,steps)
        k = (bd.arange(steps, dtype=PRECISION_TYPE) + 0.5) / steps
        k = k[None, :]  # (1,steps)

        # Broadcast to (N,steps)
        t_mid = t0[:, None] + length[:, None] * k  # (N,steps)

        # For rays without a valid segment, force t_mid=0 so we ignore them later
        t_mid = bd.where(has_seg[:, None], t_mid, bd.zeros_like(t_mid))

        # ------------------------------------------------------------------
        # Evaluate sample positions p(t) for all rays and steps
        # ------------------------------------------------------------------
        # Expand positions/directions to (N,1,3)
        pos_exp = positions[:, None, :]  # (N,1,3)
        dir_exp = directions[:, None, :]  # (N,1,3)

        # Expand t_mid to (N,steps,1)
        t_mid_exp = t_mid[..., None]  # (N,steps,1)

        p = pos_exp + t_mid_exp * dir_exp  # (N,steps,3)
        px = p[:, :, 0]
        py = p[:, :, 1]
        pz = p[:, :, 2]

        # ------------------------------------------------------------------
        # World -> image mapping, vectorized
        # ------------------------------------------------------------------
        # In your convention, camera at origin looking toward -Z, object space z<0.
        vx = px
        vy = py
        vz = pz

        # Points with vz >= 0 are outside the backward-looking FoV.
        inside = vz < 0.0

        # Angles
        theta_x = bd.arctan2(vx, -vz)
        theta_y = bd.arctan2(vy, -vz)

        u = (theta_x + half_horizontal) / horizontalAoV_rad
        v = (theta_y + half_vertical) / verticalAoV_rad

        inside = inside & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)

        # Map to floating pixel coordinates
        x_img = u * (W - 1)
        y_img = v * (H - 1)

        # ------------------------------------------------------------------
        # Bilinear sampling of alpha for all samples
        # ------------------------------------------------------------------
        # Floor to indices
        x0 = bd.floor(x_img).astype(bd.int64)
        y0 = bd.floor(y_img).astype(bd.int64)

        # Clamp to valid ranges
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

        # Gather alpha taps; alphaArray is (H,W) → [row=y, col=x]
        a00 = self.alphaArray[y0, x0]
        a10 = self.alphaArray[y0, x1]
        a01 = self.alphaArray[y1, x0]
        a11 = self.alphaArray[y1, x1]

        alpha_local = w00 * a00 + w10 * a10 + w01 * a01 + w11 * a11  # (N,steps)

        # Mask out samples outside FoV or on rays without segments
        alpha_local = bd.where(inside, alpha_local, bd.zeros_like(alpha_local))
        alpha_local = bd.where(has_seg[:, None], alpha_local, bd.zeros_like(alpha_local))

        # Ignore tiny alpha
        alpha_local = bd.where(alpha_local > alpha_eps, alpha_local, bd.zeros_like(alpha_local))

        # ------------------------------------------------------------------
        # Volumetric accumulation: T = Π (1 - alpha_step) along steps
        # ------------------------------------------------------------------
        step_T = 1.0 - alpha_local
        # Clip to [0,1] for numerical safety
        step_T = bd.clip(step_T, 0.0, 1.0)

        # Product over the step axis (axis=1) → final transmittance per ray
        T_final = bd.prod(step_T, axis=1)  # (N,)

        # Rays without segments have step_T==1 → T_final==1 → alpha_acc==0
        alpha_acc = 1.0 - T_final
        alpha_acc = bd.clip(alpha_acc, 0.0, 1.0)

        # ------------------------------------------------------------------
        # alpha_acc = probability to DROP
        # ------------------------------------------------------------------
        rnd = RNG.rand(N)
        keep_mask = rnd >= alpha_acc

        return RayBatch(incidents.value[keep_mask])


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


    def _GeneratePolarPointSources(self, appendAOV=False):
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

        if appendAOV:
            # Maybe sometimes AOV are not necessary to be included
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

        else:
            aov_cols = []

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