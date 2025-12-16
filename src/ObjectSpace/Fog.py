


from src.Raytracing.RayBatch import RayBatch
from Util.Backend import backend as bd
from Util.Globals import RNG, LambdaLines, RefreshRNG
from .Attenuator import Attenuator


class FogAttenuator(Attenuator):
    """
    Simple homogeneous fog / haze model with optional ambient 'airlight'.
    """
    def __init__(self, sl=500000, ms=1e-2,
                 extinction_length=500000,
                 ambient_level=64,
                 whitenLength = 500000,
                 maxWhiten = 2):
        """
        Parameters
        ----------
        sl : float
            scatterLength – distance scale for direction randomization.
        ms : float
            maxScatter – maximum blend toward random directions.
        extinction_length : float or None
            Distance scale for radiance extinction. If None, reuse sl.
        ambient_level : float
            Strength of ambient 'whitening' radiance to fade in at distance.
            0 means disabled.
        """
        super().__init__()
        self.scatterLength = sl
        self.maxScatter = ms
        self.extinction_length = extinction_length
        self.ambient_level = ambient_level

        self.whitenLength = whitenLength
        self.maxWhiten = maxWhiten

        self.minDistance = 8000

        # Precompute RGB lambda lines (B, G, R)
        self.RGB_LINES = bd.array([
            LambdaLines["g"],  # Blue
            LambdaLines["e"],  # Green
            LambdaLines["C'"],  # Red
        ])

    def Attenuate(self, incidents: RayBatch):
        """
        Apply fog effects to the given RayBatch.

        - For r < minDistance: identity (no fog).
        - For r >= minDistance: use r_eff = r - minDistance to drive:
            * angular scattering
            * ambient brightening (radiance)
            * spectral whitening (wavelength)
        """
        RefreshRNG()

        if incidents is None or incidents.IsNoneType():
            return incidents

        scattered = incidents.Copy()

        # --------------------------------------------------------------
        # 1. Positions, directions, distances
        # --------------------------------------------------------------
        pos  = incidents.Position()   # (N, 3)
        dirs = incidents.Direction()  # (N, 3)
        N    = pos.shape[0]

        # Raw distance from origin
        r = bd.linalg.norm(pos, axis=1)   # (N,)

        # Effective distance for fog: starts only after minDistance
        r_eff = bd.maximum(r - self.minDistance, 0.0)   # (N,)

        # If there is no effective distance AND no ambient/whitening,
        # nothing to do. But still allow ambient/whitening to be 0.
        if not bd.any(r_eff > 0):
            return scattered

        # --------------------------------------------------------------
        # 2. Directional scattering (same logic as before, but with r_eff)
        # --------------------------------------------------------------
        if self.maxScatter > 0.0:
            scatter_strength = self.maxScatter * (1.0 - bd.exp(-r_eff / self.scatterLength))

            # Sample random directions on the unit sphere
            u1 = RNG.rand(N)  # in [0, 1]
            u2 = RNG.rand(N)  # in [0, 1]

            z = 1.0 - 2.0 * u1                 # cos(theta) in [-1, 1]
            t = 2.0 * bd.pi * u2               # azimuth in [0, 2π)
            r_xy = bd.sqrt(1.0 - z * z)

            x = r_xy * bd.cos(t)
            y = r_xy * bd.sin(t)
            random_dirs = bd.stack((x, y, z), axis=1)   # (N, 3), unit vectors

            # Blend original direction with random direction, biased to small changes
            u_mix = RNG.rand(N)                # uniform in [0,1]
            mix_local = scatter_strength * (u_mix ** 2)   # (N,)
            mix_local = mix_local.reshape(-1, 1)          # (N,1)

            new_dirs = (1.0 - mix_local) * dirs + mix_local * random_dirs

            # Normalize to unit vectors
            norm = bd.linalg.norm(new_dirs, axis=1, keepdims=True)
            eps  = 1e-12
            norm = bd.maximum(norm, eps)
            new_dirs = new_dirs / norm

            scattered.SetDirection(new_dirs)

        # --------------------------------------------------------------
        # 3. Ambient brightening / airlight (radiance modulation)
        # --------------------------------------------------------------
        if self.ambient_level > 0.0:
            Lext = self.extinction_length or self.scatterLength

            # Transmittance based on effective distance
            T = bd.exp(-r_eff / Lext)      # (N,)
            T = T.reshape(-1, 1)           # (N,1) for broadcasting

            # Current polarized radiance terms: (a, tilt, b)
            rad = scattered.RadianceTerms()   # (N,3)

            # Unpolarized, white-ish ambient: same diag, zero tilt
            a_env = self.ambient_level
            b_env = self.ambient_level
            c_env = 0.0   # no tilt

            a_vec = bd.full((N,), a_env)
            c_vec = bd.full((N,), c_env)
            b_vec = bd.full((N,), b_env)

            rad_env = bd.stack((a_vec, c_vec, b_vec), axis=1)   # (N,3)

            new_rad = T * rad + (1.0 - T) * rad_env
            scattered.SetRadianceTerms(new_rad)

        # --------------------------------------------------------------
        # 4. Spectral whitening (wavelength snapping)
        # --------------------------------------------------------------
        if self.maxWhiten > 0.0:
            Lw = self.whitenLength or self.scatterLength

            # Per-ray whitening probability in [0, maxWhiten]
            whiten_strength = self.maxWhiten * (1.0 - bd.exp(-r_eff / Lw))  # (N,)

            u_whiten = RNG.rand(N)  # (N,)
            whiten_mask = u_whiten < whiten_strength  # (N,) bool

            if bd.any(whiten_mask):
                count = whiten_mask.sum()
                # CuPy → Python int if needed
                if hasattr(count, "get"):
                    count = int(count.get())
                else:
                    count = int(count)

                # Randomly choose B/G/R line indices
                choice_idx = RNG.randint(0, 3, size=count)
                new_lambda = self.RGB_LINES[choice_idx]

                # Overwrite wavelength column 6 for whitened rays
                scattered.value[whiten_mask, 6] = new_lambda

        return scattered


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================




