
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ColorWavelength import RGBToWavelengthSameD, RGBToWavelengthSpotSim, Lumi
from Util.Misc import  GridNormalized, PolarToCartesian, ArrayNormalized
from Util.PltPlot import DrawDirection, DrawPoints, DrawPointsPerColor, DrawRaybatch, SetUnifScale
from Util.Globals import ONE, INIT_ELLIPSE_TILT, FAR_DISTANCE, RNG, RefreshRNG, LambdaLines, ZERO, COLOR_PDF
from Util.ColorPDF import ColorPDF
from Raytracing.RayBatch import RayBatch




class PointsSource:
    """
    Point sources are organized in the form of:
    [[x, y, z, R, G, B], 
       [...], [...], ...]
    RGB should be float number in the range of [0, 1].
    Alternatively, it may also be using field angle representation:
    [[θ_x, θ_y, D, R, G, B], 
       [...], [...], ...]] 
    Where D is the distance from the front vertex of the lens, which in this case is the polar origin.
    """

    def __init__(self, data=None):
        self.value = data


        """Whether the data is Cartesian XYZ coordinates or field angles"""    
        self.isCartesian = True
        # This should be set to True by default. The only case polar coordinate is useful is spot testing, which can be manually entered, flagged, and adjusted. 


        """Whether the angle is in radian"""
        self.angleInRad = False


        self.sampleRecord = None


        # Whether to use probability density function for wavelength generation, or direct Fraunhofer line replacement
        self._usePDF = True


        self._ResetSampleRecord()


    def SetPoints(self, points):
        self.value = points
        self._ResetSampleRecord()


    def AddPoint(self, point):
        if(self.value is None):
            self.value = point
        else:
            bd.vstack(self.value, point)
        self._ResetSampleRecord()


    def Position(self):
        """
        Calculate and return the Cartesian coordinate of the points. If the point sources are created using field angles, i.e., polar coordinates, they will be converted to Cartesian.  
        """
        if(self.isCartesian):
            return self.value[:, :3]
        else:
            return PolarToCartesian(self.value[:, :3], self.angleInRad)
            
    
    def Color(self):
        return self.value[:, 3:6]


    def AOV(self):
        """
        Return additional AOV columns in each point row, or None if no AOV present.
        PointsSource base format is:
            [x, y, z, R, G, B, (optional AOV...)]
        """
        if self.value.shape[1] <= 6:
            return None
        return self.value[:, 6:]


    def DisplayColor(self):
        """Return an array of color that has been clipped.
        Clipping is used due to most rendering are still within [0, 1] and the only higher values are highlights that do not present much direct visual information. """
        colors = self.Color()
        return bd.clip(colors, None, 1.0)


    def EmitTowards(self,  targets, sampleCount, flareGlare=False):

        # If this method is called, it probably is from ImageSystem, so jitter and addSecondary is extremely unlikely to be used.
        return self.EmitSamplesToward(targets, sampleCount)


    def EmitSamplesToward(self, targets, sampleCount=64, jitter=None, addSecondary=None):
        """
        Emit rays from some of the point sources towards the target points. 

        :param targets: collection of points as emission targets. 
        :param sampleCount: amount of point sources to be sampled from all the soruces.

        :return: raybatch object of rays from the point sources to the target, with corresponding wavelengths. 
        """

        # In the case that there are fewer sources than demanded sample count, return all
        if(self.sampleRecord.shape[0] <= sampleCount):
            return self._SamplesToTargetsEmission(self, targets, jitter=jitter, addSecondary=addSecondary)

        # Select the ones that have been sampled the least to ensure the sample is relatively even 
        selectedIndices = self._SelectLeastSampled(sampleCount)

        # Increase the sample records of the selected 
        self.sampleRecord[selectedIndices] += 1

        # Create a new source instance using the selected points 
        sourceDuplicate = PointsSource(self.value[selectedIndices])

        # Copy the coordinate and angle setting from this one to the children
        sourceDuplicate.isCartesian = self.isCartesian
        sourceDuplicate.angleInRad = self.angleInRad

        #print("Sample min/max on master:", bd.min(self.sampleRecord), bd.max(self.sampleRecord))

        # If the sources come from a vari depth image, the jitters may be a 1D array corresponding to each source
        if (bd.ndim(jitter) == 1): jitter = jitter[selectedIndices]

        return self._SamplesToTargetsEmission(
            sourceDuplicate, 
            targets, 
            jitter, 
            addSecondary)
        

    def GenerateSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=5):
        """
        Generate white point sources convering from the axis to the postive direction of the given x and y angle. 
        """

        xAngles = bd.linspace(0, xAngle, sampleField)
        yAngles = bd.linspace(0, yAngle, sampleField)

        dists = - bd.ones_like(xAngles) * bd.array(dist)

        #append = self._PolarToCart(bd.column_stack((xAngles, yAngles, dists)))
        append = bd.column_stack((xAngles, yAngles, dists))

        # Concatenate the two arrays along axis 1 (columns)
        self.value = bd.concatenate([append, bd.full((append.shape[0], 3), 1)], axis=1)

        self._ResetSampleRecord()


    def GenerateGridSpots(self, xAngle, yAngle, dist=FAR_DISTANCE, sampleField=10):
        """
        Generate a symmetric 2D grid of white point sources spanning from 
        (-xAngle, -yAngle) to (xAngle, yAngle).
        """

        # Create symmetric angle ranges for both axes
        xAngles = bd.linspace(-xAngle, xAngle, sampleField)
        yAngles = bd.linspace(-yAngle, yAngle, sampleField)

        # Create 2D grid of (x, y) angles
        xGrid, yGrid = bd.meshgrid(xAngles, yAngles)

        # Flatten the mesh to 1D arrays
        xFlat = xGrid.ravel()
        yFlat = yGrid.ravel()

        # All distances set to a negative value (e.g., for direction into scene)
        dists = -bd.ones_like(xFlat) * dist

        # Combine into Nx3 coordinates: (xAngle, yAngle, distance)
        gridPoints = bd.column_stack((xFlat, yFlat, dists))

        # Append a white intensity vector (1, 1, 1) to each point
        fullPoints = bd.concatenate([gridPoints, bd.full((gridPoints.shape[0], 3), 1)], axis=1)

        # Store the full array
        self.value = fullPoints

        # Reset sample record
        self._ResetSampleRecord()


    def GenerateFixPoint(self, position=None, color=bd.array([1, 1, 1])):

        if(position is None):
            position = bd.array([0, 0, FAR_DISTANCE])

        assemb = bd.concatenate([position, color])
        self.value = bd.stack([assemb, assemb], axis=0)
        self._ResetSampleRecord()
        

    def GetSampleRatios(self):
        """
        This method examines the current sample record and returns the min, the max, and the ratio between min and max. The return van be used as a level of completion estimate or a sample space signal to noise ratio examination. 
        """
        minSampleCount = bd.min(self.sampleRecord)
        maxSampleCount = bd.max(self.sampleRecord)

        return minSampleCount, \
                maxSampleCount, \
                minSampleCount/maxSampleCount


    def DrawPoints(self):
        """
        Draw the points sources in 3D space with corresponding colors.
        """
        DrawPointsPerColor(self.Position(), self.Color())


    def ToString(self):
        result = "[\n"
        for row in self.value:
            # Convert each row's elements to strings and join them with commas
            row_str = "  [" + ", ".join(str(x) for x in row) + "],\n"
            result += row_str
        
        result += "]"
        return result


    def Stats(self):
        if self.value is None or self.value.shape[0] == 0:
            return "No points."

        colors = self.Color()
        if colors is None or colors.size == 0:
            return "No color data."

        def _cpu(x):
            # CuPy arrays have .get(); NumPy arrays/scalars don't.
            return x.get() if hasattr(x, "get") else x

        # Global min/max across all RGB entries
        cmin = _cpu(bd.min(colors))
        cmax = _cpu(bd.max(colors))

        # Per-channel mean/std (RGB)
        mean_rgb = _cpu(bd.mean(colors, axis=0))
        std_rgb = _cpu(bd.std(colors, axis=0))

        # Ensure plain Python floats for nicer formatting
        cmin = float(cmin)
        cmax = float(cmax)

        mr, mg, mb = (float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]))
        sr, sg, sb = (float(std_rgb[0]), float(std_rgb[1]), float(std_rgb[2]))

        return (
            "PointsSource Color Stats:\n"
            f"  Min: {cmin:.6g}\n"
            f"  Max: {cmax:.6g}\n"
            f"  Mean (R,G,B): [{mr:.6g}, {mg:.6g}, {mb:.6g}]\n"
            f"  Std  (R,G,B): [{sr:.6g}, {sg:.6g}, {sb:.6g}]"
        )


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetSampleRecord(self):
        if(self.value is None):
            return  
        else: 
            self.sampleRecord = bd.zeros(self.value.shape[0]).astype(bd.int32)


    def _SamplesToTargetsEmission(self, sampleSource, targets, jitter=None, addSecondary=None, cosineFalloff=True):

        if COLOR_PDF:
            return self._SamplesToTargetsEmissionChannelBased(sampleSource, targets, jitter, cosineFalloff)

        else:
            return self._SamplesToTargetsEmissionFraunhoferLine(sampleSource, targets, jitter, addSecondary, cosineFalloff)


    def _SamplesToTargetsEmissionFraunhoferLine(self, sampleSource, targets, jitter=None, addSecondary=None, cosineFalloff=True):
        """
        Emit rays from all samples towards all targets. This creates a cross emission of MxN size where M is the number
        of samples and N is the number of targets.
        """

        # ------------------------------------------------------------------
        # Geometry: expand source positions and directions
        # ------------------------------------------------------------------
        sourcePos = bd.copy(sampleSource.Position())
        n = sourcePos.shape[0]
        m = targets.shape[0]

        sourcePos = sourcePos[:, bd.newaxis, :]  # (n, 1, 3)
        sourcePos = bd.tile(sourcePos, (1, m, 1))  # (n, m, 3)

        if jitter is not None:
            RefreshRNG()
            if bd.ndim(jitter) == 0:  # scalar → broadcast
                mag = bd.full(sourcePos.shape[:2], float(jitter))
            elif bd.ndim(jitter) == 1:  # 1-D, per source
                if jitter.shape[0] != sourcePos.shape[0]:
                    raise ValueError("jitter 1-D length must equal number of sources")
                mag = bd.tile(jitter[:, bd.newaxis], (1, m))
            else:
                raise ValueError("Unsupported jitter dimensionality")

            mag_xyz = mag[..., bd.newaxis] * bd.array([1, 1, 0])
            jitter_off = RNG.uniform(low=-mag_xyz, high=mag_xyz)
            sourcePos += jitter_off

        targetsExpanded = targets[bd.newaxis, :, :]  # (1, m, 3)

        dirCross = targetsExpanded - sourcePos  # (n, m, 3)
        dirCross = GridNormalized(dirCross)

        # Base 6D ray data (position + direction)
        base_pd = bd.concatenate([sourcePos, dirCross], axis=2)  # (n, m, 6)

        # ------------------------------------------------------------------
        # Color → wavelength & radiant (brightness) mapping
        # ------------------------------------------------------------------
        if addSecondary is not None:
            wavelengths, radiant = RGBToWavelengthSpotSim(sampleSource.Color(), addCount=addSecondary)
        else:
            wavelengths, radiant = RGBToWavelengthSameD(sampleSource.Color())

        # wavelengths: (n, n_lambda)
        # radiant:    (n, n_lambda)  <-- may have entries > 1 now

        # Expand wavelengths to match (n, m, n_lambda)
        wavelengths = wavelengths[:, bd.newaxis, :]  # (n, 1, n_lambda)
        wavelengths = bd.tile(wavelengths, (1, m, 1))  # (n, m, n_lambda)

        wavelengthCount = wavelengths.shape[2]

        # Split wavelength cube into a list of (n, m, 1)
        wavelength_slices = bd.split(wavelengths, indices_or_sections=wavelengthCount, axis=2)

        # ------------------------------------------------------------------
        # Cosine^4 falloff mask (independent from brightness)
        # ------------------------------------------------------------------
        if cosineFalloff:
            cosTheta = bd.abs(dirCross[..., 2]) ** 4  # (n, m)
            randCos = bd.random.random(cosTheta.shape)
            angMask = randCos < cosTheta  # (n, m) boolean
        else:
            angMask = bd.ones(dirCross.shape[:2], dtype=bool)

        # ------------------------------------------------------------------
        # AOV grid (per point, per target) — same for all wavelengths
        # ------------------------------------------------------------------
        pointAOV = sampleSource.AOV()  # (n, k) or None
        aov_grid = None
        if pointAOV is not None:
            k = pointAOV.shape[1]
            aov_grid = pointAOV[:, bd.newaxis, :]  # (n, 1, k)
            aov_grid = bd.tile(aov_grid, (1, m, 1))  # (n, m, k)

        # ------------------------------------------------------------------
        # Build rays for each wavelength, allowing radiant > 1
        # and propagate AOVs in lockstep with rays
        # ------------------------------------------------------------------
        appended_all_rays = []  # list of (N_i, 7)
        appended_all_aov = []  # list of (N_i, k), only if aov_grid is not None

        for i in range(wavelengthCount):
            # Rays for this wavelength before masking: (n, m, 7)
            rays_i = bd.concatenate([base_pd, wavelength_slices[i]], axis=2)

            # Expected ray count per source for this wavelength:
            # L_j = radiant[j, i], possibly > 1
            L = radiant[:, i]  # (n,)
            base = bd.floor(L).astype(bd.int32)  # integer part
            frac = L - base  # fractional part in [0, 1)

            # Broadcast base & frac along targets dimension
            base_b = base[:, bd.newaxis]  # (n, 1)
            frac_b = frac[:, bd.newaxis]  # (n, 1)

            # Per-wavelength accumulation
            rays_list_i = []
            aov_list_i = []

            # Integer copies
            max_base = int(bd.max(base)) if base.size > 0 else 0
            if max_base > 0:
                for kcopy in range(max_base):
                    copy_mask = (base_b > kcopy) & angMask  # (n, m) boolean

                    rays_copy = rays_i[copy_mask]  # (N_k, 7)
                    if rays_copy.shape[0] == 0:
                        continue

                    rays_list_i.append(rays_copy)

                    if aov_grid is not None:
                        aov_copy = aov_grid[copy_mask]  # (N_k, k)
                        aov_list_i.append(aov_copy)

            # Fractional extra copy
            if bd.any(frac > 0):
                randFrac = bd.random.random(angMask.shape)  # (n, m)
                frac_mask = (randFrac < frac_b) & angMask  # (n, m) boolean

                rays_frac = rays_i[frac_mask]  # (N_frac, 7)
                if rays_frac.shape[0] > 0:
                    rays_list_i.append(rays_frac)
                    if aov_grid is not None:
                        aov_frac = aov_grid[frac_mask]  # (N_frac, k)
                        aov_list_i.append(aov_frac)

            # Collect for this wavelength
            if len(rays_list_i) > 0:
                appended_all_rays.append(bd.concatenate(rays_list_i, axis=0))
                if aov_grid is not None:
                    appended_all_aov.append(bd.concatenate(aov_list_i, axis=0))

        # If no rays survived (e.g., very low brightness everywhere)
        if len(appended_all_rays) == 0:
            empty = bd.zeros((0, 11))
            return RayBatch(empty)

        # Concatenate all wavelengths
        appended = bd.concatenate(appended_all_rays, axis=0)  # (#rays, 7)
        if aov_grid is not None and len(appended_all_aov) > 0:
            aov_appended = bd.concatenate(appended_all_aov, axis=0)  # (#rays, k)
        else:
            aov_appended = None

        # ------------------------------------------------------------------
        # Polarization / meta columns
        # ------------------------------------------------------------------
        temp = bd.ones(4)
        temp[0] = ONE  # Sagittal radiant
        temp[1] = ONE  # Tangential radiant
        temp[2] = INIT_ELLIPSE_TILT  # Phase difference
        temp[3] = bd.zeros_like(temp[3])  # Surface index

        core = bd.concatenate([appended, bd.tile(temp, (appended.shape[0], 1))], axis=1)

        # Append AOV data if available
        if aov_appended is not None:
            # aov_appended should match core rows exactly
            if aov_appended.shape[0] != core.shape[0]:
                raise RuntimeError(
                    f"AOV replication mismatch: aov rows={aov_appended.shape[0]}, "
                    f"core rows={core.shape[0]}"
                )
            core = bd.concatenate([core, aov_appended], axis=1)

        return RayBatch(core)


    def _SamplesToTargetsEmissionChannelBased(self, sampleSource, targets, jitter=None, cosineFalloff=True):

        # ------------------------------------------------------------------
        # Geometry: expand source positions and directions (same as _SamplesToTargetsEmission)
        # ------------------------------------------------------------------
        sourcePos = bd.copy(sampleSource.Position())
        n = sourcePos.shape[0]
        m = targets.shape[0]

        sourcePos = bd.tile(sourcePos[:, bd.newaxis, :], (1, m, 1))  # (n, m, 3)

        if jitter is not None:
            # NOTE: do NOT reseed RNG here; let it advance naturally.
            if bd.ndim(jitter) == 0:
                mag = bd.full(sourcePos.shape[:2], float(jitter))  # (n, m)
            else:
                mag = bd.tile(jitter[:, bd.newaxis], (1, m))  # (n, m)

            mag_xyz = mag[..., bd.newaxis] * bd.array([1, 1, 0])
            jitter_off = RNG.uniform(low=-mag_xyz, high=mag_xyz)
            sourcePos += jitter_off

        dirCross = GridNormalized(targets[bd.newaxis, :, :] - sourcePos)  # (n, m, 3)
        base_pd = bd.concatenate([sourcePos, dirCross], axis=2)  # (n, m, 6)

        # ------------------------------------------------------------------
        # Cosine^4 falloff mask
        # ------------------------------------------------------------------
        if cosineFalloff:
            cosTheta = bd.abs(dirCross[..., 2]) ** 4  # (n, m)
            randCos = RNG.rand(*cosTheta.shape)
            angMask = randCos < cosTheta
        else:
            angMask = bd.ones(dirCross.shape[:2], dtype=bool)

        # ------------------------------------------------------------------
        # AOV grid (per point, per target)
        # ------------------------------------------------------------------
        pointAOV = sampleSource.AOV()  # (n, k) or None
        aov_grid = None
        if pointAOV is not None:
            aov_grid = bd.tile(pointAOV[:, bd.newaxis, :], (1, m, 1))  # (n, m, k)

        # ------------------------------------------------------------------
        # Vectorized wavelength sampling + emission build (multi-sample per channel)
        # ------------------------------------------------------------------
        colorConverter = ColorPDF()
        colors = sampleSource.Color()  # (n, 3)

        # No per-point normalization. Use raw RGB as expected multiplier.
        wR = bd.maximum(colors[:, 0] * bd.array(colorConverter.normGainR), ZERO)
        wG = bd.maximum(colors[:, 1] * bd.array(colorConverter.normGainG), ZERO)
        wB = bd.maximum(colors[:, 2] * bd.array(colorConverter.normGainB), ZERO)

        muR = LambdaLines[colorConverter.lineR]
        muG = LambdaLines[colorConverter.lineG]
        muB = LambdaLines[colorConverter.lineB]

        # meta columns (radiance stays ONE; we increase sample count instead)
        temp = bd.ones(4)
        temp[0] = ONE  # Sagittal radiant
        temp[1] = ONE  # Tangential radiant
        temp[2] = INIT_ELLIPSE_TILT
        temp[3] = bd.zeros_like(temp[3])

        blocks = []

        def _append_core_from_flat(base_flat, lam_flat, ch_idx, aov_flat=None):
            # base_flat: (N,6), lam_flat: (N,)
            rays7 = bd.concatenate([base_flat, lam_flat[:, bd.newaxis]], axis=1)
            core11 = bd.concatenate([rays7, bd.tile(temp, (rays7.shape[0], 1))], axis=1)
            ch_col = bd.full((core11.shape[0], 1), bd.array(float(ch_idx)), dtype=core11.dtype)
            core12 = bd.concatenate([core11, ch_col], axis=1)
            if aov_flat is not None:
                core12 = bd.concatenate([core12, aov_flat], axis=1)
            blocks.append(core12)

        def _emit_integer_sheet(keep_src_mask, lam_per_src, ch_idx):
            # This keeps your existing integer-copy behavior:
            # one wavelength per source, replicated across all targets, masked by angMask.
            if not bd.any(keep_src_mask):
                return

            base_c = base_pd[keep_src_mask]  # (ns, m, 6)
            mask_c = angMask[keep_src_mask]  # (ns, m)
            ns = base_c.shape[0]

            base_flat = bd.reshape(base_c, (ns * m, 6))
            mask_flat = bd.reshape(mask_c, (ns * m,))
            base_flat = base_flat[mask_flat]

            lam_rep = bd.repeat(lam_per_src[keep_src_mask], m)
            lam_rep = lam_rep[mask_flat]

            aov_flat = None
            if aov_grid is not None:
                aov_c = aov_grid[keep_src_mask]
                aov_flat = bd.reshape(aov_c, (ns * m, aov_c.shape[2]))
                aov_flat = aov_flat[mask_flat]

            _append_core_from_flat(base_flat, lam_rep, ch_idx, aov_flat)

        def _emit_fractional_nm(frac_mask_nm, lam_nm, ch_idx):
            # Fraunhofer-like fractional behavior:
            # per (source,target) mask AND per (source,target) wavelength.
            if not bd.any(frac_mask_nm):
                return

            base_flat = base_pd[frac_mask_nm]  # (N,6)
            lam_flat = lam_nm[frac_mask_nm]  # (N,)

            aov_flat = None
            if aov_grid is not None:
                aov_flat = aov_grid[frac_mask_nm]  # (N,k)

            _append_core_from_flat(base_flat, lam_flat, ch_idx, aov_flat)

        # Safety cap to prevent extreme HDR values (e.g., 1000+) from exploding memory.
        MAX_COPIES_PER_SRC = 256

        def _emit_weighted(w, mu, sigma, ch_idx):
            k = bd.floor(w).astype(int)
            frac = w - bd.array(k, dtype=w.dtype)

            # Determine loop count on CPU (works for numpy/cupy)
            kmax = bd.max(k)
            if hasattr(kmax, "get"):
                kmax = int(kmax.get())
            else:
                kmax = int(kmax)

            kmax = min(kmax, MAX_COPIES_PER_SRC)

            # Integer copies (unchanged)
            for rep in range(kmax):
                keep = k > rep
                lam = bd.clip(bd.array(mu) + bd.array(sigma) * RNG.randn(n), 380.0, 780.0)
                _emit_integer_sheet(keep, lam, ch_idx)

            # Fractional copy (CHANGED to per-(n,m), Fraunhofer-like)
            # - per-target decision avoids coherent "sheet" speckles
            # - per-target wavelength avoids identical-wavelength pupil sheets
            frac_b = bd.clip(frac[:, bd.newaxis], ZERO, ONE)  # (n,1)
            randFrac = RNG.rand(n, m)  # (n,m)
            frac_mask_nm = (randFrac < frac_b) & angMask  # (n,m)

            lam_nm = bd.clip(
                bd.array(mu) + bd.array(sigma) * RNG.randn(n, m),
                380.0, 780.0
            )
            _emit_fractional_nm(frac_mask_nm, lam_nm, ch_idx)

        _emit_weighted(wR, muR, colorConverter.sigmaR, 0)
        _emit_weighted(wG, muG, colorConverter.sigmaG, 1)
        _emit_weighted(wB, muB, colorConverter.sigmaB, 2)

        if len(blocks) == 0:
            col_count = 12 + (pointAOV.shape[1] if pointAOV is not None else 0)
            return RayBatch(bd.zeros((0, col_count)))

        return RayBatch(bd.concatenate(blocks, axis=0))


    def _SelectLeastSampled(self, sampleCount):
        """
        Find all indices with the least amount of sample history.
        """

        min_count = bd.min(self.sampleRecord)
        candidates = bd.where(self.sampleRecord == min_count)[0]
        
        # Shuffle candidates to break ties randomly
        RNG.shuffle(candidates)
        
        if len(candidates) >= sampleCount:
            return candidates[:sampleCount]
        
        # Get remaining points sorted by sample count
        sorted_indices = bd.argsort(self.sampleRecord)
        remaining_indices = sorted_indices[len(candidates):]
        
        # Shuffle remaining indices before selection
        RNG.shuffle(remaining_indices)
        
        selected = bd.concatenate([candidates, remaining_indices[:sampleCount - len(candidates)]])
        return selected[:sampleCount]
    



def main():
    

    t = PointsSource(bd.array(
        [[1., 1., 0, 1, 1, 1], 
         [-1, 1., 0, 1, 1, 1], 
         [-1, -1, 0, 1, 1, 1], 
         [1., -1, 0, 1, 1, 1]]
    ))
    t = PointsSource()
    t.GenerateFixPoint(bd.array([0, 0, -12]))

    targets = bd.array(
        [[1., 1., 4], 
         [-1, 1., 4], 
         [-1, -1, 4], 
         [1., -1, 4]]
    )
    print(t.Position())
    #DrawPoints(t.Position(), ptSize=1)
    DrawPoints(targets, ptSize=1)

    for i in range(10):
        pos = []
        RB = t.EmitSamplesToward(targets, sampleCount=16, jitter=1)
        DrawRaybatch(RB, arrowRatio=0, lLength=4.75)
        pos.append(RB.Position())
        DrawPoints(RB.Position(), ptSize=4)

    
    SetUnifScale(5)
    plt.show()
    
    



if __name__ == "__main__":
    main() 

