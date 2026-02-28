

from .Film import Film
from Util.Backend import backend as bd


class Technicolor(Film):
    """
    Technicolor-like dye-transfer abstraction:
      - keep Film's spectral response weighting (_ApplyColorPDF) unchanged (inherited)
      - disable dye clouds entirely
      - each grain's "density/value" is directly mapped from that channel's exposure
      - final RGB is assembled ONLY from the three rasterized grain layers
        (no multiplicative overlay on the original RGB image)
    """

    def __init__(self, sr=None, bfd=42, w=36, h=24, horiPx=1920):
        super().__init__(sr=sr, bfd=bfd, w=w, h=h, horiPx=horiPx)

        self.silverGrainsPerMP = 2560000.0
        self.grainSizeMu = [2.5, 2.5, 2.5]
        self.grainSizeSigma = [.25, .25, .25]
        self.silverAmax = 2.0

        # ------------------------------------------------------------
        # Turn off Film "print stock" behaviors that are not desired
        # ------------------------------------------------------------
        # Not used because we override ApplyGrainAndNoise, but keep safe defaults.
        self.bleachByPassRatio = 1.0
        self.dyeCloudStrength = 0.0
        self.dyeCloudPhotonScale = 0.0

        # ------------------------------------------------------------
        # Technicolor density mapping controls (art-directable, fixed)
        # ------------------------------------------------------------
        # Exposure -> grain density mapping:
        #   density = (E / (E + ref))^gamma  in [0,1]
        # "ref" is your "half-sat" exposure level (linear units).
        self.tcRefExposure = 0.1

        # Shaping: <1 boosts shadows, >1 compresses shadows.
        self.tcGamma = 0.85

        # Optional per-channel multipliers (acts like layer "printer lights")
        self.tcChannelGain = [1.0, 1.0, 1.0]

        # Minor stochastic fluctuation per grain (not per pixel).
        # This is *not* Poisson-y film grain; it’s intentionally subtle.
        self.tcGrainJitterSigma = 0.03  # typical 0.01~0.06

        # Underexposure wash-away (stop bath): below tcToeExposure => no grains
        self.tcToeExposure = 0.002  # in your linear exposure units; tune this
        self.tcToeWidth = 0.003  # soft transition width; 0 for hard cutoff
        self.tcToePower = 1.0  # >1 makes toe steeper, <1 softer

        # If True: output intensity = 1 - density (i.e., treat density as absorption).
        # If False: output intensity = density (density “denotes intensity” directly).
        self.tcInvert = False

        # Optional small blur/clumping on the rasterized fields (keeps “transfer” feel)
        self.tcClump = False
        self.tcClumpRadiusPx = 1
        self.tcClumpStrength = 0.85

        # Final clamp
        self.tcClampMin = 0.0
        self.tcClampMax = 100



    # ------------------------------------------------------------
    # Public pipeline hook: return ONLY the three grain layers
    # ------------------------------------------------------------
    def ApplyGrainAndNoise(self, rgb_image):
        """
        Override Film.ApplyGrainAndNoise:
          - no dye clouds
          - no overlay on the original RGB
          - final = stack(r_layer, g_layer, b_layer)
        """
        return self._ApplyTechnicolorGrains(rgb_image)

    # ------------------------------------------------------------
    # Core technicolor grains
    # ------------------------------------------------------------
    def _ApplyTechnicolorGrains(self, rgb_image):
        rgb = bd.asarray(rgb_image)
        rgb = bd.maximum(rgb, 0)

        H, W = rgb.shape[0], rgb.shape[1]

        # Ensure grids exist / match resolution
        if (self.GrainGridR is None) or (self.GrainGridG is None) or (self.GrainGridB is None):
            self._GenerateGrains(rgb_image)
        if int(self._grain_grid_H) != int(H) or int(self._grain_grid_W) != int(W):
            self._GenerateGrains(rgb_image)

        # Develop grain values directly from exposure (per layer)
        gR = self._develop_grains_from_exposure_tc(rgb[..., 0], self.GrainGridR, ch=0)
        gG = self._develop_grains_from_exposure_tc(rgb[..., 1], self.GrainGridG, ch=1)
        gB = self._develop_grains_from_exposure_tc(rgb[..., 2], self.GrainGridB, ch=2)

        # Rasterize via your existing power-Voronoi / polygonal border logic
        fieldR = self._rasterize_power_voronoi(gR, H, W)
        fieldG = self._rasterize_power_voronoi(gG, H, W)
        fieldB = self._rasterize_power_voronoi(gB, H, W)

        # Optional mild clumping/softening (more “transfer” than “noisy emulsion”)
        if self.tcClump :
            rad = self.tcClumpRadiusPx
            if rad > 0:
                s = self.tcClumpStrength
                fieldR = bd.maximum(fieldR, self._box_blur_2d(fieldR, rad) * s)
                fieldG = bd.maximum(fieldG, self._box_blur_2d(fieldG, rad) * s)
                fieldB = bd.maximum(fieldB, self._box_blur_2d(fieldB, rad) * s)

        out = bd.stack((fieldR, fieldG, fieldB), axis=-1)

        if self.tcInvert:
            out = 1.0 - out

        # max_val = self.tcClampMax
        # out = bd.clip(out, float(self.tcClampMin), max_val)
        # return out
        # Prevent division by zero at pure 1.0
        out_safe = bd.clip(out, 0.0, 0.9999)

        # Undo the gamma and the E / (E + ref) mapping
        out_linear = bd.power(out_safe, 1.0 / self.tcGamma)
        out = (out_linear * self.tcRefExposure) / (1.0 - out_linear)

        return bd.maximum(out, float(self.tcClampMin))


    def _develop_grains_from_exposure_tc(self, exposure2d, grid, ch: int):
        """
        Technicolor “development” with stop-bath wash-away:
          - below toe => grains = 0
          - jitter is also suppressed in the toe region so black stays clean
        """
        exp2d = bd.asarray(exposure2d)
        exp2d = bd.maximum(exp2d, 0)

        x = grid[:, 0]
        y = grid[:, 1]
        sens = grid[:, 3]

        # Sample exposure at grain centers
        E = self._bilinear_sample(exp2d, x, y)

        # Channel gain (printer lights)
        try:
            g = float(self.tcChannelGain[ch])
        except Exception:
            g = 1.0
        E = E * g

        # ---------- Stop-bath wash-away gate (toe) ----------
        toe = float(max(getattr(self, "tcToeExposure", 0.0), 0.0))
        width = float(max(getattr(self, "tcToeWidth", 0.0), 0.0))

        if toe > 0.0:
            if width <= 0.0:
                # Hard cutoff: anything below toe is fully washed away
                gating = (E >= toe).astype(bd.float32)
            else:
                # Smoothstep gate between [toe, toe+width]
                t = (E - toe) / max(width, 1e-12)
                t = bd.clip(t, 0.0, 1.0)
                gating = (t * t * (3.0 - 2.0 * t)).astype(bd.float32)
        else:
            gating = bd.ones_like(E, dtype=bd.float32)

        toe_pow = float(max(getattr(self, "tcToePower", 1.0), 1e-6))
        if abs(toe_pow - 1.0) > 1e-6:
            gating = bd.power(gating, toe_pow)

        # ---------- Exposure -> density mapping ----------
        ref = float(max(getattr(self, "tcRefExposure", 0.18), 1e-12))
        base = E / (E + ref)

        gamma = float(max(getattr(self, "tcGamma", 1.0), 1e-6))
        base = bd.power(bd.clip(base, 0.0, 1.0), gamma)

        # Apply wash-away gate *before* jitter so underexposed grains are truly dead
        base = base * gating

        # ---------- Minor per-grain jitter (suppressed in toe) ----------
        sig = float(max(getattr(self, "tcGrainJitterSigma", 0.03), 0.0))
        if sig > 0:
            sens_tame = bd.power(bd.clip(sens, 0.25, 4.0), 0.35)

            # Key: jitter scaled by gating so black doesn't get re-activated by noise
            jitter = bd.random.standard_normal(base.shape) * sig * sens_tame * gating
            base = base + jitter

        #a = bd.clip(base, 0.0, 1.0)
        a = bd.maximum(base, 0.0)

        # Optional: enforce exact zeros in hard-toe region even after numeric fuzz
        # (useful if width>0 but you still want "truly empty" very dark)
        # a = a * (gating > 0).astype(bd.float32)

        grid[:, 4] = a.astype(bd.float32)
        return grid