from matplotlib.pyplot import scatter

from src.Raytracing.RayBatch import RayBatch
from Util.Backend import backend as bd
from Util.Globals import RNG, LambdaLines



class Attenuator:
    def __init__(self):
        """
        """

    def Attenuate(self, incidents:RayBatch):
        pass



class DepthVisualizer(Attenuator):
    """
    Pseudo Z-depth visualizer.

    - Quantizes wavelengths into three Fraunhofer lines representing base RGB:
        * B: LambdaLines["g"]
        * G: LambdaLines["e"]
        * R: LambdaLines["C'"]
    - Given a [min_distance, max_distance] range, randomly drops rays with a
      probability that increases toward max_distance.

    Typical usage:
        dv = DepthVisualizer(min_distance=..., max_distance=...)
        rays_out = dv.Attenuate(rays_in)
    """

    def __init__(self, min_distance: float = None, max_distance: float = None):
        super().__init__()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # Cache base RGB wavelengths
        self._rgb_lambda = bd.array([
            LambdaLines["g"],   # Blue
            LambdaLines["e"],   # Green
            LambdaLines["C'"],  # Red
        ], dtype=bd.float64)

    def Attenuate(self, incidents: RayBatch):
        if incidents is None or incidents.IsNoneType():
            return incidents

        # Work on a copy so we don't mutate the input raybatch in-place
        rays = incidents.Copy()

        # ------------------------------------------------------------------
        # 1. Get positions and distances (for depth)
        # ------------------------------------------------------------------
        pos = rays.Position()              # (N, 3)
        N = pos.shape[0]
        if N == 0:
            return rays

        dist = bd.linalg.norm(pos, axis=1)  # (N,)

        # Determine depth range
        d_min = self.min_distance if self.min_distance is not None else dist.min()
        d_max = self.max_distance if self.max_distance is not None else dist.max()

        # Normalize / sanity check
        if d_max <= d_min:
            # Degenerate range: no depth variation, just recolor and return
            self._assign_pseudo_rgb_wavelengths(rays)
            return rays

        # ------------------------------------------------------------------
        # 2. Assign pseudo-RGB wavelengths (discrete Fraunhofer lines)
        # ------------------------------------------------------------------
        self._assign_pseudo_rgb_wavelengths(rays)

        # ------------------------------------------------------------------
        # 3. Distance-dependent random culling
        # ------------------------------------------------------------------
        # Normalize depth to [0, 1] where 0 = near (d_min), 1 = far (d_max)
        eps = 1e-12
        t = (dist - d_min) / (d_max - d_min + eps)
        t = bd.clip(t, 0.0, 1.0)  # just in case of numerical noise

        # Drop probability increases toward the far plane:
        #   p_drop = t  (near -> 0, far -> 1)
        p_drop = t

        # Sample uniform random numbers to decide which rays to keep
        u = RNG.rand(N)  # (N,), in [0, 1)
        keep_mask = u > p_drop

        # Safety: if we accidentally drop all rays, keep at least the nearest one
        if not keep_mask.any():
            nearest_idx = int(dist.argmin())
            keep_mask[nearest_idx] = True


        rays = rays.Mask(keep_mask)


        return rays


    def ColorizeDepthZones(self, incidents: RayBatch,
                           near_distance: float,
                           far_distance: float):
        """
        Divide rays into 3 depth zones based on ||position|| and assign
        a *single* wavelength per zone from the RGB Fraunhofer lines:

            zone 1 (nearest):   LambdaLines["C'"]  (R)
            zone 2 (middle):    LambdaLines["e"]   (G)
            zone 3 (farthest):  LambdaLines["g"]   (B)

        Parameters
        ----------
        incidents : RayBatch
            Input rays.
        near_distance : float
            First depth boundary.
        far_distance : float
            Second depth boundary. If < near_distance, they are swapped.

        Returns
        -------
        RayBatch
            A *copy* of the input raybatch with wavelengths quantized
            per zone.
        """
        if incidents is None or incidents.IsNoneType():
            return incidents

        # Work on a copy
        rays = incidents.Copy()

        pos = rays.Position()  # (N, 3)
        N = pos.shape[0]
        if N == 0:
            return rays

        # Euclidean distance from origin
        dist = bd.linalg.norm(pos, axis=1)  # (N,)

        # Ensure d1 <= d2
        d1 = float(near_distance)
        d2 = float(far_distance)
        if d2 < d1:
            d1, d2 = d2, d1

        # Fixed RGB Fraunhofer lines:
        lambda_near = LambdaLines["C'"]  # Red
        lambda_mid = LambdaLines["e"]  # Green
        lambda_far = LambdaLines["g"]  # Blue

        # Build masks for the three zones
        mask_near = dist <= d1
        mask_mid = (dist > d1) & (dist <= d2)
        mask_far = dist > d2

        # Assign a single wavelength per zone
        # Uses the same convention as _assign_pseudo_rgb_wavelengths:
        #   wavelength stored in value[:, 6]
        wl = rays.value[:, 6]

        wl[mask_near] = lambda_near
        wl[mask_mid] = lambda_mid
        wl[mask_far] = lambda_far

        rays.value[:, 6] = wl

        return rays


    # ==============================================================
    """ ===================== Calculations ===================== """
    # ==============================================================


    def _assign_pseudo_rgb_wavelengths(self, rays: RayBatch):
        """
        Quantize each ray's wavelength to one of three RGB Fraunhofer lines.

        The mapping is random, with equal probability for B, G, and R.
        """
        pos = rays.Position()
        N = pos.shape[0]

        # Sample indices in {0, 1, 2} and map to [λ_B, λ_G, λ_R]
        idx = RNG.randint(0, 3, size=N)
        new_lambda = self._rgb_lambda[idx]

        # Try the most likely wavelength API first
        rays.value[:, 6] = new_lambda




