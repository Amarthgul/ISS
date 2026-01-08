

from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import LambdaLines

Number = Union[int, float]

@dataclass
class SensorSpec:
    """Physical imager properties used to map PSF samples (meters) -> pixels."""
    width_mm: float
    height_mm: float
    res_x: int
    res_y: int

    @property
    def pixel_pitch_x_m(self) -> float:
        return (self.width_mm * 1e-3) / float(self.res_x)

    @property
    def pixel_pitch_y_m(self) -> float:
        return (self.height_mm * 1e-3) / float(self.res_y)



class Diffraction:
    """
    Diffraction star / PSF generator for a photographic imaging pipeline.

    Core idea:
      - Use an amplitude pupil from a diaphragm/pupil alpha image
      - Optionally apply field-dependent pupil distortion + vignetting (amplitude)
      - Optionally apply defocus via quadratic phase (phase)
      - Compute coherent PSF via FFT (Fraunhofer at the focused image plane)
      - Map the PSF’s native sample spacing (≈ λ * f/#) to sensor pixels
      - Convolve only a highlight layer and add back

    """

    def __init__(
        self,
        pupilImage,
        sensor: Optional[SensorSpec] = None,
        kernel_radius_px: int = 256,
    ):
        """
        :param pupilImage: B&W alpha image (2D). Values in [0,1] where 1 = open.
        :param sensor: SensorSpec for mapping PSF meters -> pixels. Required for physically sized PSFs.
        :param kernel_radius_px: Half-size (radius) of the PSF kernel to return/apply in pixels.
        """

        # -------------------- Inputs --------------------
        self.pupil = self._to_float_array(pupilImage)  # 2D array in [0,1]
        self._pupil_h, self._pupil_w = int(self.pupil.shape[0]), int(self.pupil.shape[1])

        # -------------------- Optical parameters --------------------
        self.focalLength_mm: float = 50.0
        self.fNumber: float = 8.0

        # Assume flat field: sensor plane offset from ideal focus plane (mm). Positive = sensor moved away.
        self.deFocusDist_mm: float = 0.0

        # RGB representative wavelengths (meters); default uses provided LambdaLines.
        self.RGB_nm: List[float] = [
            float(LambdaLines["r"]) * 1e-9,
            float(LambdaLines["e"]) * 1e-9,
            float(LambdaLines["g"]) * 1e-9
        ]
        # Convert to meters on access:
        # If LambdaLines already stores meters, user can override via SetRGBWavelengthsMeters.

        # -------------------- Sensor spec --------------------
        self.sensor: Optional[SensorSpec] = sensor

        # -------------------- Field-dependent pupil approximation knobs --------------------
        # Simple foreshortening model: scale pupil along x by cos(theta_x), along y by cos(theta_y)
        self.enable_field_foreshorten: bool = True

        # Simple pupil shift model (in normalized pupil radius units): shift = k * tan(theta)
        # e.g. k=0.05 means at 20 deg, shift≈0.018 radii.
        self.pupil_shift_k: float = 0.0

        # Simple vignetting clip: shrink effective pupil radius with field angle
        # effective_radius = 1 - v_k * (theta/theta_max)^2 (clamped)
        self.enable_field_vignetting: bool = False
        self.vignetting_k: float = 0.0
        self.vignetting_theta_max_deg: float = 30.0

        # -------------------- Output / application knobs --------------------
        self.kernel_radius_px: int = int(kernel_radius_px)

        # Highlight extraction: use luminance weights in linear RGB
        self.luma_weights = bd.array([0.2126, 0.7152, 0.0722])

        # Cache: PSFs keyed by (fieldAngle tuple, wavelength samples tuple, defocus, fnum, f)
        self._psf_cache = {}

    # ==================================================================
    # Public API
    # ==================================================================

    def SetSensor(self, width_mm: float, height_mm: float, res_x: int, res_y: int):
        self.sensor = SensorSpec(width_mm=width_mm, height_mm=height_mm, res_x=int(res_x), res_y=int(res_y))
        self._psf_cache.clear()


    def SetOptics(self, focalLength_mm: float, fNumber: float, deFocusDist_mm: float = 0.0):
        self.focalLength_mm = float(focalLength_mm)
        self.fNumber = float(fNumber)
        self.deFocusDist_mm = float(deFocusDist_mm)
        self._psf_cache.clear()


    def SetRGBWavelengthsMeters(self, wavelengths_m: Tuple[float, float, float]):
        """Override representative wavelengths for (R,G,B) in meters."""
        self.RGB_nm = [w * 1e9 for w in wavelengths_m]
        self._psf_cache.clear()


    def ApplyDiffraction(self, targetImage, highlightThreshold: float, fieldAngle_deg: Number = 0.0):
        """
        Clip out then convolve the highlight part of the image and add it back in.

        :param targetImage: RGB image to be processed, shape (H,W,3). Expected in linear space [0,1] or HDR.
        :param highlightThreshold: threshold in normalized luminance (0..1-ish) to extract highlights.
        :param fieldAngle_deg: for now uses a single PSF (space-invariant) at this field angle. For piecewise,
                              call PSF(...) per tile and blend externally.
        :return: processed RGB image, same shape as targetImage.
        """
        img = self._to_float_array(targetImage)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("targetImage must have shape (H,W,3).")

        # Extract highlight layer
        lum = img[..., 0] * self.luma_weights[0] + img[..., 1] * self.luma_weights[1] + img[..., 2] * self.luma_weights[2]
        mask = lum > float(highlightThreshold)
        highlight = img * mask[..., None]

        # Get a single RGB PSF kernel
        psf_rgb = self.PSF([float(fieldAngle_deg)], wavelengthInterp=0)[0]  # (K,K,3)
        # Convolve highlight with PSF (per channel)
        out = img.copy()
        for c in range(3):
            conv = self._fft_convolve2d(highlight[..., c], psf_rgb[..., c])
            out[..., c] = out[..., c] + conv

        return out


    def PSF(self, fieldAngles: List[Union[Number, Tuple[Number, Number]]], wavelengthInterp: int = 1):
        """
        Calculates the PSF kernels (RGB) based on the pupil image and approximations.

        :param fieldAngles: list of field angles in degrees.
            - Each entry may be:
              * scalar θ meaning (θx=θ, θy=0) i.e. horizontal field
              * tuple (θx, θy)
        :param wavelengthInterp: number of interpolations between each RGB wavelength.
            - 0 -> only 3 wavelengths (R,G,B)
            - 1 -> adds one intermediate between each pair (total 5)
            - n -> adds n intermediates between each neighboring pair
        :return: list of PSF kernels, each with shape (K,K,3), normalized so sum over kernel ≈ 1 per channel.
        """

        # wavelength samples for each channel (meters)
        lam_rgb_m = [self._nm_to_m(x) for x in self.RGB_nm]
        lam_samples_m = self._build_wavelength_samples(lam_rgb_m, int(wavelengthInterp))

        results = []
        for ang in fieldAngles:
            if isinstance(ang, (list, tuple)) and len(ang) == 2:
                theta_x, theta_y = float(ang[0]), float(ang[1])
            else:
                theta_x, theta_y = float(ang), 0.0

            cache_key = (
                round(theta_x, 6),
                round(theta_y, 6),
                tuple(round(l * 1e9, 4) for l in lam_samples_m),
                round(self.focalLength_mm, 6),
                round(self.fNumber, 6),
                round(self.deFocusDist_mm, 6),
                self.kernel_radius_px,
                (self.sensor.width_mm, self.sensor.height_mm, self.sensor.res_x, self.sensor.res_y),
            )
            if cache_key in self._psf_cache:
                results.append(self._psf_cache[cache_key])
                continue

            # Build pupil function (amplitude + phase)
            A = self._AmplitudePupilPSF(theta_x, theta_y)  # 2D float, [0,1]
            # PSF RGB accumulation
            kernel_rgb = bd.zeros((self.kernel_radius_px * 2 + 1, self.kernel_radius_px * 2 + 1, 3), dtype=A.dtype)

            # For each channel, integrate over wavelength samples with simple weights:
            # We use triangular weights assigning each sample to nearest channel to keep it simple and stable.
            # You can replace this with spectral -> XYZ -> RGB if you want.
            for lam in lam_samples_m:
                weights_rgb = self._simple_rgb_weights(lam, lam_rgb_m)
                if weights_rgb is None:
                    continue

                phi = self._PhasePupilPSF(theta_x, theta_y, lam_m=lam)  # 2D float phase (radians)
                P = A * bd.exp(1j * phi)

                # Coherent PSF at focal plane (Fraunhofer)
                # Field at image plane is FFT of pupil function; intensity is |.|^2
                E = self._fftshift(self._fft2(self._ifftshift(P)))
                I = bd.abs(E) ** 2

                # Map native focal-plane sampling (meters) to sensor pixels and crop to kernel radius
                kernel_mono = self._resample_psf_to_sensor_pixels(I, lam_m=lam)

                # Accumulate weighted into RGB
                for c in range(3):
                    kernel_rgb[..., c] = kernel_rgb[..., c] + kernel_mono * weights_rgb[c]

            # Normalize each channel energy to 1 (so convolution conserves that channel’s highlight energy)
            kernel_rgb = self._normalize_kernel_rgb(kernel_rgb)

            self._psf_cache[cache_key] = kernel_rgb
            results.append(kernel_rgb)

        return results


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _AmplitudePupilPSF(self, theta_x_deg: float, theta_y_deg: float):
        """
        Create field-dependent amplitude pupil A(x,y) from the diaphragm alpha image.

        This models: foreshortening, shift, and optional vignetting clip.
        """
        A = self.pupil

        # Normalize and clamp
        A = bd.clip(A, 0.0, 1.0)

        if not (self.enable_field_foreshorten or self.pupil_shift_k != 0.0 or self.enable_field_vignetting):
            return A

        # Build normalized coordinates in pupil image space: u,v in [-1,1]
        H, W = self._pupil_h, self._pupil_w
        yy, xx = self._meshgrid_norm(H, W)  # yy,xx in [-1,1]

        # Foreshortening
        sx = 1.0
        sy = 1.0
        if self.enable_field_foreshorten:
            sx = max(0.2, float(bd.cos(bd.array(theta_x_deg * bd.pi / 180.0))))
            sy = max(0.2, float(bd.cos(bd.array(theta_y_deg * bd.pi / 180.0))))

        # Shift
        shx = self.pupil_shift_k * float(bd.tan(bd.array(theta_x_deg * bd.pi / 180.0)))
        shy = self.pupil_shift_k * float(bd.tan(bd.array(theta_y_deg * bd.pi / 180.0)))

        # Inverse map sampling coordinates (u',v') -> original (u,v)
        u = (xx - shx) / sx
        v = (yy - shy) / sy

        # Optional vignetting clip (radial shrink)
        if self.enable_field_vignetting and self.vignetting_k > 0.0:
            rdeg = (theta_x_deg ** 2 + theta_y_deg ** 2) ** 0.5
            t = min(1.0, max(0.0, rdeg / float(self.vignetting_theta_max_deg)))
            rad = max(0.0, 1.0 - self.vignetting_k * (t ** 2))
        else:
            rad = 1.0

        # Outside effective pupil -> 0
        rr = bd.sqrt(u * u + v * v)
        inside = rr <= float(rad)

        # Sample original A at (u,v) via bilinear in image coords
        sampled = self._bilinear_sample(A, u, v)
        sampled = sampled * inside

        return sampled


    def _PhasePupilPSF(self, theta_x_deg: float, theta_y_deg: float, lam_m: float):
        """
        Approximate pupil phase (radians). Minimum model: defocus only.
        """
        # Normalized pupil radius rho in [0,1]
        H, W = self._pupil_h, self._pupil_w
        yy, xx = self._meshgrid_norm(H, W)
        rho = bd.sqrt(xx * xx + yy * yy)
        rho = bd.clip(rho, 0.0, 1.0)

        # Defocus mapping (paraxial, flat field)
        # W20 (waves) ≈ Δz / (8 λ N^2), phase = 2π W20 rho^2
        dz_m = float(self.deFocusDist_mm) * 1e-3
        N = float(self.fNumber)
        W20 = dz_m / (8.0 * float(lam_m) * (N ** 2) + 1e-30)
        phi = 2.0 * bd.pi * W20 * (rho ** 2)

        # NOTE: You can add coma/astig terms here if desired, driven by fieldAngles.
        return phi


    # -------------------- Wavelength / RGB weights --------------------

    def _build_wavelength_samples(self, lam_rgb_m: List[float], interp: int) -> List[float]:
        """Build a small sampled set of wavelengths between the RGB representative points."""
        if interp <= 0:
            return list(lam_rgb_m)

        # Ensure increasing order for sampling; keep mapping order for weights separately
        # But for a typical RGB set (R>G>B in wavelength), sort descending by wavelength.
        # We'll sample across the full span and then weight into RGB by proximity.
        lam_min = min(lam_rgb_m)
        lam_max = max(lam_rgb_m)
        steps = 2 + 2 * interp  # for two intervals
        # Use a modest uniform sampling across [min,max]
        return [lam_min + (lam_max - lam_min) * (i / float(steps - 1)) for i in range(steps)]


    def _simple_rgb_weights(self, lam: float, lam_rgb_m: List[float]):
        """
        Assign a wavelength sample into RGB weights based on proximity to the representative wavelengths.
        This is a pragmatic approximation to avoid a full CMF integration.
        """
        # If user supplied wavelengths in any order, just use nearest neighbor with soft weights.
        d = [abs(lam - lr) for lr in lam_rgb_m]
        # Invert distance with epsilon
        inv = [1.0 / (di + 1e-12) for di in d]
        s = sum(inv)
        if s <= 0:
            return None
        w = [v / s for v in inv]
        return w


    # -------------------- PSF mapping to sensor pixels --------------------


    def _resample_psf_to_sensor_pixels(self, I_focal, lam_m: float):
        """
        Map the FFT-computed focal-plane intensity grid to a sensor-pixel kernel.

        The FFT output corresponds to samples in focal plane with spacing approximately:
            Δx_focal = λ * f / D = λ * N  (meters)
        where N=f/#, D=f/N.

        We then resample to pixel pitch and crop to kernel_radius_px.
        """
        # Native focal-plane sample pitch in meters
        dx_native = float(lam_m) * float(self.fNumber)  # meters per sample

        # Desired pixel pitch (assume square pixels; if not, use x and y separately)
        px = float(self.sensor.pixel_pitch_x_m)
        py = float(self.sensor.pixel_pitch_y_m)

        # Use average pitch for kernel sampling (keeps kernel square).
        p = 0.5 * (px + py)

        # Crop around center of I_focal first (keeps resampling small)
        I = I_focal
        H, W = int(I.shape[0]), int(I.shape[1])
        cy, cx = H // 2, W // 2

        # Choose an intermediate crop radius in native samples that covers desired kernel
        # desired physical radius on sensor:
        r_phys = float(self.kernel_radius_px) * p
        r_native = int(max(8, min(min(H, W) // 2 - 2, (r_phys / dx_native) * 1.25)))
        y0, y1 = cy - r_native, cy + r_native + 1
        x0, x1 = cx - r_native, cx + r_native + 1
        I_crop = I[y0:y1, x0:x1]

        # Now resample I_crop from dx_native spacing to pixel spacing p
        scale = dx_native / p  # >1 means native samples are coarser than pixels
        # We want output size ~ (2*kernel_radius_px+1)
        out_size = int(self.kernel_radius_px * 2 + 1)
        kernel = self._resample_centered_square(I_crop, in_scale=1.0, out_size=out_size, out_scale=scale)

        # Ensure nonnegative and normalize energy
        kernel = bd.clip(kernel, 0.0, None)
        s = float(bd.sum(kernel))
        if s > 0:
            kernel = kernel / s
        return kernel


    # -------------------- Convolution --------------------

    def _fft_convolve2d(self, image2d, kernel2d):
        """
        FFT-based *linear* convolution (same output size as image).
        Zero-pads to avoid circular wrap-around artifacts.
        """
        img = image2d
        ker = kernel2d

        H, W = int(img.shape[0]), int(img.shape[1])
        kh, kw = int(ker.shape[0]), int(ker.shape[1])

        # Pad to size needed for linear convolution
        P = H + kh - 1
        Q = W + kw - 1

        # Zero-pad image (place in top-left)
        img_pad = bd.zeros((P, Q), dtype=img.dtype)
        img_pad[:H, :W] = img

        # Zero-pad kernel (place in top-left)
        # FIX: Do NOT ifftshift here. Just place the kernel normally.
        ker_pad = bd.zeros((P, Q), dtype=ker.dtype)
        ker_pad[:kh, :kw] = ker

        # Convolve in frequency domain
        F_img = self._fft2(img_pad)
        F_ker = self._fft2(ker_pad)
        out_full = bd.real(self._ifft2(F_img * F_ker))

        # FIX: The result is now shifted by the kernel center (kh//2, kw//2).
        # We must crop the valid middle region.
        start_y = kh // 2
        start_x = kw // 2

        return out_full[start_y: start_y + H, start_x: start_x + W]

    # -------------------- Helpers --------------------

    def _normalize_kernel_rgb(self, k_rgb):
        out = k_rgb
        for c in range(3):
            s = float(bd.sum(out[..., c]))
            if s > 0:
                out[..., c] = out[..., c] / s
        return out

    def _to_float_array(self, x):
        """Convert input to backend float array."""
        arr = bd.array(x)
        # Promote to float
        if str(arr.dtype).startswith("int") or str(arr.dtype).startswith("uint"):
            arr = arr.astype(bd.float32)
        else:
            arr = arr.astype(bd.float32) if arr.dtype != bd.float32 and arr.dtype != bd.float64 else arr
        return arr

    def _nm_to_m(self, lam_nm_or_m: float) -> float:
        # If value looks like nanometers (hundreds), convert to meters.
        if lam_nm_or_m > 1e-3:
            return lam_nm_or_m * 1e-9
        return lam_nm_or_m

    def _meshgrid_norm(self, H: int, W: int):
        """Return (yy, xx) normalized to [-1,1] with pixel centers."""
        y = (bd.arange(H, dtype=bd.float32) + 0.5) / float(H) * 2.0 - 1.0
        x = (bd.arange(W, dtype=bd.float32) + 0.5) / float(W) * 2.0 - 1.0
        yy, xx = bd.meshgrid(y, x, indexing="ij")
        return yy, xx

    def _bilinear_sample(self, img2d, u, v):
        """
        Bilinear sample img2d where u,v are normalized coords in [-1,1].
        Output has same shape as u/v.
        """
        H, W = int(img2d.shape[0]), int(img2d.shape[1])

        # Map to pixel coordinates
        x = (u + 1.0) * 0.5 * (W - 1)
        y = (v + 1.0) * 0.5 * (H - 1)

        x0 = bd.floor(x).astype(bd.int32)
        y0 = bd.floor(y).astype(bd.int32)
        x1 = bd.clip(x0 + 1, 0, W - 1)
        y1 = bd.clip(y0 + 1, 0, H - 1)

        x0 = bd.clip(x0, 0, W - 1)
        y0 = bd.clip(y0, 0, H - 1)

        wx = (x - x0.astype(bd.float32))
        wy = (y - y0.astype(bd.float32))

        Ia = img2d[y0, x0]
        Ib = img2d[y0, x1]
        Ic = img2d[y1, x0]
        Id = img2d[y1, x1]

        wa = (1.0 - wx) * (1.0 - wy)
        wb = wx * (1.0 - wy)
        wc = (1.0 - wx) * wy
        wd = wx * wy

        return Ia * wa + Ib * wb + Ic * wc + Id * wd

    def _resample_centered_square(self, img2d, in_scale: float, out_size: int, out_scale: float):
        """
        Resample a centered square image to out_size x out_size.
        - img2d is assumed centered at its middle pixel.
        - out_scale maps output pixel spacing relative to input pixel spacing:
            coordinate_in = coordinate_out * out_scale
        This uses bilinear sampling.
        """
        H, W = int(img2d.shape[0]), int(img2d.shape[1])
        cy, cx = (H - 1) * 0.5, (W - 1) * 0.5

        # Output grid coordinates centered at 0 in "output pixel units"
        o = (bd.arange(out_size, dtype=bd.float32) - (out_size - 1) * 0.5)
        yy, xx = bd.meshgrid(o, o, indexing="ij")

        # Map to input pixel coords
        xin = xx * float(out_scale) + cx
        yin = yy * float(out_scale) + cy

        # Convert to normalized [-1,1] in input image space
        u = (xin / (W - 1)) * 2.0 - 1.0
        v = (yin / (H - 1)) * 2.0 - 1.0

        return self._bilinear_sample(img2d, u, v)

    # -------------------- Backend FFT helpers --------------------

    def _fft2(self, x):
        fft = getattr(bd, "fft", None)
        if fft is None:
            raise RuntimeError("Backend does not provide bd.fft")
        return fft.fft2(x)

    def _ifft2(self, x):
        fft = getattr(bd, "fft", None)
        if fft is None:
            raise RuntimeError("Backend does not provide bd.fft")
        return fft.ifft2(x)

    def _fftshift(self, x):
        fft = getattr(bd, "fft", None)
        if fft is None:
            raise RuntimeError("Backend does not provide bd.fft")
        return fft.fftshift(x)

    def _ifftshift(self, x):
        fft = getattr(bd, "fft", None)
        if fft is None:
            raise RuntimeError("Backend does not provide bd.fft")
        return fft.ifftshift(x)
