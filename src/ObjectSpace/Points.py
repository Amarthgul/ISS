

import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ColorWavelength import RGBToWavelengthSameD, RGBToWavelengthSpotSim, Lumi
from Util.Misc import  GridNormalized, PolarToCartesian, ArrayNormalized
from Util.PltPlot import DrawDirection, DrawPoints, DrawPointsPerColor, DrawRaybatch, SetUnifScale
from Util.Globals import ONE, INIT_ELLIPSE_TILT, FAR_DISTANCE, RNG, RefreshRNG
from Raytracing.RayBatch import RayBatch




class PointsSource:
    """
    Point sources are organized in the form of:
    [[x, y, z, R, G, B], 
       [...], [...], ...]
    RGB must be float number in the range of [0, 1]. 
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


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ResetSampleRecord(self):
        if(self.value is None):
            return  
        else: 
            self.sampleRecord = bd.zeros(self.value.shape[0]).astype(bd.int32)

    def _SamplesToTargetsEmission(self, sampleSource, targets, jitter=None, addSecondary=None, cosineFalloff=True):
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
            wavelengths, radiants = RGBToWavelengthSpotSim(sampleSource.Color(), addCount=addSecondary)
        else:
            wavelengths, radiants = RGBToWavelengthSameD(sampleSource.Color())

        # wavelengths: (n, n_lambda)
        # radiants:    (n, n_lambda)  <-- may have entries > 1 now

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
        # Build rays for each wavelength, allowing radiants > 1
        # and propagate AOVs in lockstep with rays
        # ------------------------------------------------------------------
        appended_all_rays = []  # list of (N_i, 7)
        appended_all_aov = []  # list of (N_i, k), only if aov_grid is not None

        for i in range(wavelengthCount):
            # Rays for this wavelength before masking: (n, m, 7)
            rays_i = bd.concatenate([base_pd, wavelength_slices[i]], axis=2)

            # Expected ray count per source for this wavelength:
            # L_j = radiants[j, i], possibly > 1
            L = radiants[:, i]  # (n,)
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

