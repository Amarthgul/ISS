

import matplotlib.pyplot as plt

from Surfaces.Surface import FieldStopType, Surface
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.PltPlot import Reset2D, DrawPlane
from Util.Globals import INFINITY, ZERO, ONE, TWO, Axis, OUTPUT_TYPE, COLOR_PDF
from Util.ColorWavelength import WavelengthToRGB
from Util.Misc import PointsInTriangle, NumpyConversion


class StdImager(Surface):
    """
    Standard virtual imager with no optical effect. 
    """
    def __init__(self, bfd = 42, w = 36, h = 24, horiPx = 1920):

        # Raduis is infinity, no thickness, clear semi-diameter is infinity
        super().__init__(INFINITY, ZERO, INFINITY)

        self.fType = FieldStopType.Rectangular

        #self.rayBatch = None 
        self.BFD = bfd  # Back focal distance. Sensor distance from last element's vertex 
        
        self.width = bd.array(w)          # Physical size in mm
        self.height = bd.array(h)         # Physical size in mm    

        self.horizontalPx = horiPx  # Unitless int 
        self.verticalPx = None      # Unitless int 

        """Points reprensenting the 4 corners of the rectangular field stop"""
        self.gatePoints = None

        self.rayPath = None 


        # If gate points are not set, then assume it is flat and perpendicular to the optical axis, then use the 2 following properties to calculate the gate points
        self._lensLength = 0 # Length of the lens in front of the imager
        self._zPos = 0


        self._Start()


    def _Start(self):
        if (self.verticalPx is None):
            # If vertical pixel is not set, calculate it from aspect ratio
            self.verticalPx = int((self.height / self.width ) * self.horizontalPx)

        
    def SetLensLength(self, length):
        """
        Set the length of the lens from first vertex to the last vertex, this info is used to calculate the cumulative thickness of the imager z position. 

        :param length: length of the lens. 
        """
        self._lensLength = length

        self.Update()


    def Update(self):
        self._zPos = self._lensLength + self.BFD

        if (self.gatePoints is None):
            self.gatePoints = bd.array([
                [self.width / 2, self.height / 2, self._zPos],
                [-self.width / 2, self.height / 2, self._zPos],
                [-self.width / 2, -self.height / 2, self._zPos],
                [self.width / 2, -self.height / 2, self._zPos]])

        self.frontVertex = bd.array([ZERO, ZERO, self._zPos])


    def IntersectRays(self, raybatch):
        #self.rayBatch = raybatch
        intersections, _tir, _vig = self.Intersection(raybatch)
        raybatch.Mask(~_vig)
        raybatch.SetPosition(intersections)
        

        return raybatch, _tir, _vig


    def IntegralRays(self, raybatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):

        #self.rayBatch = raybatch

        return self._integralRays(raybatch, baseImg=baseImg, overExpNoiseRemoval=overExpNoiseRemoval, polarized=polarized)


    def DrawSurface(self):
        DrawPlane(self.gatePoints, color='black')


    def AcquireEmpty(self, dataType=OUTPUT_TYPE):
        """
        Accquire an empty image array. When converted, this will be a black and blank image. 
        """
        
        imgAry = bd.zeros((self.horizontalPx , self.verticalPx, 3),  dtype=dataType)
    
        return imgAry





    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _FieldStopMask(self, intersections):
        """
        Given the intersections, filter out the ones that are outside of the field stop. 
        """
        # If tilt shift is involved, consider using the triangle intersection check 
        heightMask = (intersections[:, Axis.Y.value] > self.height/2 ) |(intersections[:, Axis.Y.value] < -self.height/2)
        widthMask = (intersections[:, Axis.X.value] > self.width/2 ) | (intersections[:, Axis.X.value] < -self.width/2)

        return ~(heightMask | widthMask)


    def _integralRays(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):

        if COLOR_PDF:
            return self._integralRaysChannelBased(intersectRayBatch, baseImg, overExpNoiseRemoval,polarized)

        else:
            return self._integralRaysFraunhoferLine(intersectRayBatch, baseImg, overExpNoiseRemoval, polarized)


    def _integralRaysFraunhoferLine(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """
        Taking integral over the rays arriving at the image plane. 

        :param bitDepth: image bitdepth.
        :param plotResult: whether to show the resulting plot or not. 
        :param baseImg: if not null, the generated image will be added onto this base image. 
        :param overExpNoiseRemoval: sometimes there are single pixel of value that is way overexposed due to random errors. When overExpNoiseRemoval is set to a number, it will prune the values depending on the std of the values. 
        
        :return: a float array representing an RGB image.  
        """


        #print("Total intersect rays ", intersectRayBatch.value.shape)

        pxPitch = self.width / self.horizontalPx 
        pxOffset = bd.array([self.horizontalPx/2, self.verticalPx/2, 0])

        # Find the rays that arrived at the the image plane 
        rayHitMask = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Translate the intersections from 3D image space to 2D pixel-based space
        rayPos = intersectRayBatch.Position()[rayHitMask] / pxPitch + pxOffset
        rayWavelength = intersectRayBatch.Wavelength()[rayHitMask]

        # Convert ray position into pixel position 
        rayPos = bd.floor(rayPos).astype(int)
        # Create pixel grid 
        radiantGridR = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridG = bd.zeros( (self.horizontalPx, self.verticalPx) )
        radiantGridB = bd.zeros( (self.horizontalPx, self.verticalPx) )

        # Isolate the rays that arrived at the imager plane 
        rayHitIsolate = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Find all wavelengths 
        wavelengths = bd.unique(rayWavelength)
        for wavelength in wavelengths:
            RGB = WavelengthToRGB(wavelength)

            # Isolate the wavelength currently dealing with 
            wavelengthIsolate = bd.isclose(intersectRayBatch.Wavelength(), wavelength)

            # Convert the position of the ray hits into an int grid by flooring them 
            rayPosChannel = bd.floor(
                intersectRayBatch.Position()[rayHitIsolate & wavelengthIsolate] / pxPitch + pxOffset).astype(int)[:, :2]
            

            radiantChannel = intersectRayBatch.PolarizedRadiance(polarized)[rayHitIsolate & wavelengthIsolate]

            # Masking out the rays outside of imager area, avoiding the numpy negative index from shifting the rays...
            in_bounds = (
                    (rayPosChannel[:, 0] >= 0) & (rayPosChannel[:, 0] < self.horizontalPx) &
                    (rayPosChannel[:, 1] >= 0) & (rayPosChannel[:, 1] < self.verticalPx)
            )
            rayPosChannel = rayPosChannel[in_bounds]
            radiantChannel = radiantChannel[in_bounds]

            #print("radiantChannel max: ", bd.max(radiantChannel), "\t\t mean", bd.mean(radiantChannel), "\t\t std ", bd.std(radiantChannel))

            # Try to remove the outlier over-exposed pixels (maybe caused by float error?)
            if((overExpNoiseRemoval is not None) and 
               (bd.max(radiantChannel) > bd.mean(radiantChannel))):
                # TODO: add a second condition for automatic check if both mean and max are the same
                radiantChannel = self._PruneHighOutliers(radiantChannel, overExpNoiseRemoval)

            rChannel = radiantChannel * RGB[0]
            gChannel = radiantChannel * RGB[1]
            bChannel = radiantChannel * RGB[2]

            bd.add.at(radiantGridR, (rayPosChannel[:, 0], rayPosChannel[:, 1]), rChannel)
            bd.add.at(radiantGridG, (rayPosChannel[:, 0], rayPosChannel[:, 1]), gChannel)
            bd.add.at(radiantGridB, (rayPosChannel[:, 0], rayPosChannel[:, 1]), bChannel)
        
        # Stack the channels together as a "latent image"
        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)
        
        # Monte Carlo addition 
        if(baseImg is not None):
            rgb_image = baseImg + rgb_image

        return rgb_image


    def _integralRaysChannelBased(self, intersectRayBatch, baseImg=None, overExpNoiseRemoval=12, polarized=True):
        """
        Channel-based integral:
          - Uses rayBatch.Channel() (0=R, 1=G, 2=B) to directly deposit energy
          - Uses PolarizedRadiance(polarized=...) for intensity (same as _integralRays)
          - Optionally prunes over-exposed outliers (same idea as _integralRays)
          - Optionally adds onto baseImg (Monte Carlo accumulation)

        Extension hooks (no-ops in StdImager, override in subclasses):
          - _ApplyIncidentRayEffects: modify per-ray (rayPos/radiant/chan/wavelength) before deposition
          - _ApplyColorPDF: modify per-ray radiance using wavelength/channel (Film-like behavior)
          - _ApplyHalation: image-domain effect needing incident information (e.g. halation bloom)
          - _ApplyGrainAndNoise: image-domain noise/grain effect needing radiance image
        """

        pxPitch = self.width / self.horizontalPx
        pxOffset = bd.array([self.horizontalPx / 2, self.verticalPx / 2, 0])

        # Rays that arrived at the image plane
        rayHitMask = bd.isclose(intersectRayBatch.value[:, 2], self._zPos)

        # Convert hit positions to pixel coords
        rayPos = bd.floor(intersectRayBatch.Position()[rayHitMask] / pxPitch + pxOffset).astype(int)[:, :2]

        # Per-ray radiance + channel id + wavelength
        radiant = intersectRayBatch.PolarizedRadiance(polarized)[rayHitMask]
        chan = intersectRayBatch.Channel()[rayHitMask].astype(int)
        wavelength = intersectRayBatch.Wavelength()[rayHitMask]

        # Mask out hits outside the imager area (avoid negative indexing issues)
        in_bounds = (
                (rayPos[:, 0] >= 0) & (rayPos[:, 0] < self.horizontalPx) &
                (rayPos[:, 1] >= 0) & (rayPos[:, 1] < self.verticalPx)
        )
        rayPos = rayPos[in_bounds]
        radiant = radiant[in_bounds]
        chan = chan[in_bounds]
        wavelength = wavelength[in_bounds]

        # =================================================================================================
        # Hook: allow subclasses to alter rays before deposition (e.g. CFA shift, lateral scattering, etc.)
        rayPos, radiant, chan, wavelength = self._ApplyIncidentRayEffects(
            intersectRayBatch=intersectRayBatch,
            rayPos=rayPos,
            radiant=radiant,
            chan=chan,
            wavelength=wavelength
        )

        # =================================================================================================
        # Hook: spectral / channel weighting (default identity in StdImager)
        radiant = self._ApplyColorPDF(
            intersectRayBatch=intersectRayBatch,
            rayPos=rayPos,
            radiant=radiant,
            wavelength=wavelength,
            chan=chan
        )

        # Create pixel grids
        radiantGridR = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridG = bd.zeros((self.horizontalPx, self.verticalPx))
        radiantGridB = bd.zeros((self.horizontalPx, self.verticalPx))

        # Deposit per channel (and prune outliers per-channel, like the wavelength loop did)
        for c, grid in ((0, radiantGridR), (1, radiantGridG), (2, radiantGridB)):
            m = (chan == c)
            if not bd.any(m):
                continue

            pos_c = rayPos[m]
            rad_c = radiant[m]

            # Try to remove the outlier over-exposed rays (same condition style as _integralRays)
            if (overExpNoiseRemoval is not None) and (bd.max(rad_c) > bd.mean(rad_c)):
                # print(self._IncidentStats(intersectRayBatch))
                rad_c = self._PruneHighOutliers(rad_c, overExpNoiseRemoval)

            bd.add.at(grid, (pos_c[:, 0], pos_c[:, 1]), rad_c)

        # Stack to RGB image
        rgb_image = bd.stack((radiantGridR, radiantGridG, radiantGridB), axis=-1)

        # =================================================================================================
        # Hook: image-domain effects that may require incident info (e.g. halation)
        rgb_image = self._ApplyHalation(
            intersectRayBatch=intersectRayBatch,
            rgb_image=rgb_image
        )

        # =================================================================================================
        # Hook: image-domain noise/grain
        rgb_image = self._ApplyGrainAndNoise(
            rgb_image=rgb_image
        )

        # Monte Carlo accumulation
        if baseImg is not None:
            rgb_image = baseImg + rgb_image

        return rgb_image


    def _ApplyIncidentRayEffects(self, intersectRayBatch, rayPos, radiant, chan, wavelength):
        """Hook for per-ray manipulations before deposition.

        Use this for effects that need the incident rays, e.g.:
          - CFA-induced spatial color shift / channel mixing
          - lateral scattering / pre-blur approximations
          - per-ray clipping / masking based on direction / angle, etc.

        Must return (rayPos, radiant, chan, wavelength).
        """
        return rayPos, radiant, chan, wavelength


    def _ApplyColorPDF(self, intersectRayBatch, rayPos, radiant, wavelength, chan):
        """Hook for spectral/channel weighting (Film override point).

        Default: identity (no spectral weighting).
        Must return radiant (same shape as input).
        """
        return radiant


    def _ApplyHalation(self, intersectRayBatch, rgb_image):
        """Hook for image-domain halation / bloom-like effects.

        Default: identity.
        Must return rgb_image.
        """
        return rgb_image


    def _ApplyGrainAndNoise(self, rgb_image):
        """Hook for image-domain grain/noise.

        Default: identity.
        Must return rgb_image.
        """
        return rgb_image


    def _PruneHighOutliers(self, arr, k=4.0):
        """
        Replace values from an array that are greater than mean + k * std.
        
        :param arr: Input array.
        :param k: Number of standard deviations above the mean to use as threshold.
            
        :return: Array whose outliers are replaced with 0.
        """
        arr = bd.asarray(arr)
        mean = bd.mean(arr)
        std = bd.std(arr)
        threshold = mean + k * std
        arr[arr >= threshold] = 0
        
        return arr


    def _IncidentStats(self, intersectRayBatch):
        """
        Examine whether some rays are disrupted during propagation.

        Reports:
          - basic sanity checks (NaN/Inf) on position, direction, wavelength
          - polarization term stats (Φ, i_Φ, b)
          - validity of the 2x2 polarization ellipse matrix [[Φ, b],[b, i_Φ]]
            using SPD checks (a>0, d>0, det>0)
          - a fast "polarized radiance" proxy computed from closed-form eigenvalues
            (avoids bd.linalg.eigh to keep this diagnostic cheap)

        Returns a multi-line string (safe to print).
        """

        rb = intersectRayBatch
        if rb is None or getattr(rb, "value", None) is None:
            return "[IncidentStats] RayBatch is None."

        val = rb.value
        if val.shape[0] == 0:
            return "[IncidentStats] RayBatch is empty."

        # ---------- helpers ----------
        def _to_cpu(x):
            # works for numpy scalars/arrays and cupy
            try:
                return x.get()
            except Exception:
                return x

        def _scalar(x):
            # convert backend scalar to python float
            x = _to_cpu(x)
            try:
                return float(x)
            except Exception:
                # fallback for 0-d arrays
                return float(getattr(x, "item", lambda: x)())

        def _istat(arr):
            arr = bd.asarray(arr)
            # Note: avoid errors for all-invalid arrays by masking finite values
            finite = bd.isfinite(arr)
            if not bd.any(finite):
                return {"min": None, "max": None, "mean": None, "std": None, "finite_n": 0}
            a = arr[finite]
            return {
                "min": _scalar(bd.min(a)),
                "max": _scalar(bd.max(a)),
                "mean": _scalar(bd.mean(a)),
                "std": _scalar(bd.std(a)),
                "finite_n": int(_scalar(bd.sum(finite))),
            }

        n = val.shape[0]

        # ---------- core fields ----------
        pos = val[:, 0:3]
        direc = val[:, 3:6]
        wl = val[:, 6]

        # polarization terms: Φ (col 7), i_Φ (col 8), b (col 9)
        Phi = val[:, 7]
        iPhi = val[:, 8]
        b = val[:, 9]

        # channel (optional, but your RayBatch defines it at col 11)
        chan = None
        if val.shape[1] > 11:
            try:
                chan = val[:, 11].astype(int)
            except Exception:
                chan = None

        # ---------- finiteness checks ----------
        pos_bad = ~bd.isfinite(pos).all(axis=1)
        dir_bad = ~bd.isfinite(direc).all(axis=1)
        wl_bad = ~bd.isfinite(wl)

        pol_bad = ~(bd.isfinite(Phi) & bd.isfinite(iPhi) & bd.isfinite(b))

        bad_any = pos_bad | dir_bad | wl_bad | pol_bad

        # ---------- polarization matrix validity / near-singularity ----------
        # Matrix M = [[a, c],[c, d]] with a=Phi, d=iPhi, c=b
        a = Phi
        d = iPhi
        c = b

        # SPD conditions (necessary & sufficient for symmetric 2x2):
        # a > 0, d > 0, det = a*d - c^2 > 0
        det = a * d - c * c

        eps_det = 1e-12
        eps_eig = 1e-12

        neg_diag = (a <= 0) | (d <= 0)
        bad_det = det <= eps_det

        # closed-form eigenvalues for symmetric 2x2
        tr = a + d
        disc = bd.sqrt((a - d) * (a - d) + 4 * c * c)

        # eigenvalues: (tr +/- disc)/2
        eig1 = (tr + disc) / 2
        eig2 = (tr - disc) / 2
        min_eig = bd.minimum(eig1, eig2)

        near_singular = min_eig <= eps_eig

        pol_invalid = pol_bad | neg_diag | bad_det | near_singular

        # ---------- radiance proxy (same structure as RayBatch.PolarizedRadiance) ----------
        # semi-axis = 1/sqrt(eig); radiance = (semi1 + semi2)/2
        # clamp eigenvalues to avoid inf/nan from numerical noise
        eig1c = bd.maximum(eig1, eps_eig)
        eig2c = bd.maximum(eig2, eps_eig)
        rad_proxy = (1 / bd.sqrt(eig1c) + 1 / bd.sqrt(eig2c)) / 2

        rad_stats = _istat(rad_proxy)
        phi_stats = _istat(Phi)
        iphi_stats = _istat(iPhi)
        b_stats = _istat(b)
        det_stats = _istat(det)
        mineig_stats = _istat(min_eig)

        # ---------- counts ----------
        def _count(mask):
            return int(_scalar(bd.sum(mask)))

        cnt_pos_bad = _count(pos_bad)
        cnt_dir_bad = _count(dir_bad)
        cnt_wl_bad = _count(wl_bad)
        cnt_pol_bad = _count(pol_bad)
        cnt_bad_any = _count(bad_any)

        cnt_neg_diag = _count(neg_diag)
        cnt_bad_det = _count(bad_det)
        cnt_near_sing = _count(near_singular)
        cnt_pol_invalid = _count(pol_invalid)

        # ---------- formatting ----------
        lines = []
        lines.append(f"[IncidentStats] backend={backend_name}, rays={n}")
        lines.append(
            f"  non-finite: pos={cnt_pos_bad}, dir={cnt_dir_bad}, wl={cnt_wl_bad}, pol_terms={cnt_pol_bad}, any={cnt_bad_any}"
        )
        lines.append(
            f"  pol-matrix issues: neg_diag={cnt_neg_diag}, det<=eps={cnt_bad_det}, min_eig<=eps={cnt_near_sing}, invalid_any={cnt_pol_invalid}"
        )

        def _fmt_stat(name, s):
            if s["finite_n"] == 0:
                return f"  {name}: (no finite values)"
            return (f"  {name}: min={s['min']:.6g}, max={s['max']:.6g}, "
                    f"mean={s['mean']:.6g}, std={s['std']:.6g}, finite_n={s['finite_n']}")

        lines.append(_fmt_stat("radiance_proxy", rad_stats))
        lines.append(_fmt_stat("Phi(Φ)", phi_stats))
        lines.append(_fmt_stat("iPhi(i_Φ)", iphi_stats))
        lines.append(_fmt_stat("tilt(b)", b_stats))
        lines.append(_fmt_stat("det(Φ*i_Φ-b^2)", det_stats))
        lines.append(_fmt_stat("min_eig", mineig_stats))

        # Optional: per-channel radiance proxy stats (if channel exists)
        if chan is not None:
            for c_id, cname in ((0, "R"), (1, "G"), (2, "B")):
                m = (chan == c_id)
                if bd.any(m):
                    s = _istat(rad_proxy[m])
                    lines.append(_fmt_stat(f"radiance_proxy[{cname}]", s))

        return "\n".join(lines)







