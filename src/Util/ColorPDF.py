
"""
Probability Density Function for color-wavelength conversion.

"""


from Util.Backend import backend as bd
from Util.Globals import RNG, RefreshRNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines
from Util.MathFunctions import Erf, SkewNormPDF


def sample_skew_normal(mu, sigma, alpha, shape):
    # Sampling helper method

    # Fast path: plain Gaussian
    if alpha == 0:
        lam = bd.array(mu) + bd.array(sigma) * RNG.randn(*shape)
        return bd.clip(lam, 380.0, 780.0)

    # Azzalini skew-normal sampling:
    # mu + sigma*(delta*|z0| + sqrt(1-delta^2)*z1)
    z0 = RNG.randn(*shape)
    z1 = RNG.randn(*shape)

    a = bd.array(alpha)
    delta = a / bd.sqrt(ONE + a * a)

    w = delta * bd.abs(z0) + bd.sqrt(ONE - delta * delta) * z1
    lam = bd.array(mu) + bd.array(sigma) * w
    return bd.clip(lam, 380.0, 780.0)



class ColorPDF:

    def __init__(self):
        # These Fraunhofer lines are also the mu of the Gaussian distribution
        self.lineR = "r"
        self.lineG = "e"
        self.lineB = "g"

        # Sigma parameter for the three Gaussian respectively
        self.sigmaR = 45
        self.sigmaG = 30
        self.sigmaB = 40

        # Alpha set is for skewed Gaussian, when set to 0 they're just standard Gaussian
        self.alphaR = -2
        self.alphaG = 0
        self.alphaB = 2

        # Gain for each distribution
        # This is not used in wavelength to color conversion, since it is almost always the case that the input image is white balanced at a neutral color
        self.gainR = 1
        self.gainG = 1
        self.gainB = 1

        self.normGainR = 1
        self.normGainG = 1
        self.normGainB = 1

        # Gaussian distribution at the given wavelengths tend to reduce the max value to around 0.13-0.2, which may disrupt the existing radiance expectations, so a global scalar multiple is applied for spectral resposne.
        self._unitScalar = 60.

        self._Update()


    def ColorToWavelength(self, colors, perChannelSample=4, fastGaussian=True):
        """
        Convert many colors into a batch of wavelengths.

        :param colors: array of RGB colors in shape (m, 3), values in range [0, inf]
        :param fastGaussian: whether to ignore per channel sample and skew. When enabled, directly use a standard Gaussian with per channel sample as 1.
        :param perChannelSample: number of samples per color channel.

        :return: array of wavelengths in shape (k, 2), with:
                 col 0 = wavelength (nm)
                 col 1 = channel index (0=R, 1=G, 2=B)
        """
        if fastGaussian:
            return self._FastGaussian(colors)

        m = colors.shape[0]

        # 1) Normalize colors row-wise into [0,1]
        row_max = bd.max(colors, axis=1, keepdims=True)
        denom = bd.maximum(row_max, ONE)
        colors_n = bd.clip(colors / denom, ZERO, ONE)

        muR, muG, muB = LambdaLines[self.lineR], LambdaLines[self.lineG], LambdaLines[self.lineB]

        # ------------------------------------------------------------------
        #    Keep probability includes normalized gain:
        #    lower gain => more pruning (fewer emitted wavelengths in that channel)
        # ------------------------------------------------------------------
        pR = bd.clip(colors_n[:, 0] * bd.array(self.normGainR), ZERO, ONE)
        pG = bd.clip(colors_n[:, 1] * bd.array(self.normGainG), ZERO, ONE)
        pB = bd.clip(colors_n[:, 2] * bd.array(self.normGainB), ZERO, ONE)

        # ------------------------------------------------------------------
        #    General path: perChannelSample > 1
        # ------------------------------------------------------------------
        lamR = sample_skew_normal(muR, self.sigmaR, self.alphaR, (m, perChannelSample))
        lamG = sample_skew_normal(muG, self.sigmaG, self.alphaG, (m, perChannelSample))
        lamB = sample_skew_normal(muB, self.sigmaB, self.alphaB, (m, perChannelSample))

        # Broadcast keep probability across the sample dimension
        keepR = RNG.rand(m, perChannelSample) < pR[:, bd.newaxis]
        keepG = RNG.rand(m, perChannelSample) < pG[:, bd.newaxis]
        keepB = RNG.rand(m, perChannelSample) < pB[:, bd.newaxis]

        selR = lamR[keepR]
        selG = lamG[keepG]
        selB = lamB[keepB]

        def pack(lams_1d, ch_idx):
            if lams_1d.size == 0:
                return None
            ch = bd.full((lams_1d.shape[0],), bd.array(ch_idx), dtype=lams_1d.dtype)
            return bd.stack([lams_1d, ch], axis=1)

        arrR = pack(selR, 0.0)
        arrG = pack(selG, 1.0)
        arrB = pack(selB, 2.0)

        parts = [a for a in (arrR, arrG, arrB) if a is not None]
        if len(parts) == 0:
            return bd.zeros((0, 2), dtype=bd.float64)

        return bd.concatenate(parts, axis=0)


    def SpectralResponse(self, wavelength, channel):
        """
        Return scalars representing spectral responses intensity of a given wavelength assigned to a given channel.

        :param channel: an array of channel indices, size (m).
        :param wavelength: an array of wavelength, size (m).

        :return: an array of spectral responses, size (m).
        """

        # Output
        resp = bd.zeros_like(wavelength, dtype=bd.float64)

        # Channel masks
        mR = (channel == 0)
        mG = (channel == 1)
        mB = (channel == 2)

        # Evaluate per-channel skew-normal pdf and apply *non-normalized* gains
        if bd.any(mR):
            resp[mR] = SkewNormPDF(wavelength[mR], LambdaLines[self.lineR], self.sigmaR, self.alphaR) * bd.array(
                self.gainR)

        if bd.any(mG):
            resp[mG] = SkewNormPDF(wavelength[mG], LambdaLines[self.lineG], self.sigmaG, self.alphaG) * bd.array(
                self.gainG)

        if bd.any(mB):
            resp[mB] = SkewNormPDF(wavelength[mB], LambdaLines[self.lineB], self.sigmaB, self.alphaB) * bd.array(
                self.gainB)

        return resp * self._unitScalar


    def PlotDistribution(self):
        # Draw the three skewed gaussian distribution using the line, the sigma, and alpha
        # Color them in RGB respectively

        import numpy as np
        import matplotlib.pyplot as plt

        # Visible-ish range
        x = np.linspace(380.0, 780.0, 2000)

        muR = float(LambdaLines[self.lineR])
        muG = float(LambdaLines[self.lineG])
        muB = float(LambdaLines[self.lineB])

        sigR = float(self.sigmaR)
        sigG = float(self.sigmaG)
        sigB = float(self.sigmaB)

        aR = float(self.alphaR)
        aG = float(self.alphaG)
        aB = float(self.alphaB)

        yR = SkewNormPDF(x, muR, sigR, aR) * self.normGainR
        yG = SkewNormPDF(x, muG, sigG, aG) * self.normGainG
        yB = SkewNormPDF(x, muB, sigB, aB) * self.normGainB

        plt.figure()
        plt.plot(x, yR, color=(1.0, 0.0, 0.0), label=f"R: μ={muR:.2f}nm σ={sigR:g} α={aR:g}")
        plt.plot(x, yG, color=(0.0, 1.0, 0.0), label=f"G: μ={muG:.2f}nm σ={sigG:g} α={aG:g}")
        plt.plot(x, yB, color=(0.0, 0.0, 1.0), label=f"B: μ={muB:.2f}nm σ={sigB:g} α={aB:g}")

        # Optional: mark mus
        plt.axvline(muR, color=(1.0, 0.0, 0.0), linestyle="--", linewidth=1, alpha=0.4)
        plt.axvline(muG, color=(0.0, 1.0, 0.0), linestyle="--", linewidth=1, alpha=0.4)
        plt.axvline(muB, color=(0.0, 0.0, 1.0), linestyle="--", linewidth=1, alpha=0.4)

        plt.xlim(380.0, 780.0)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("PDF")
        plt.title("Channel Skew-Normal Wavelength PDFs")
        plt.legend()
        plt.show()


    def WavelengthVis(self, inputWavelengths):

        PlotWavelengthHistogram(inputWavelengths)


    def _Update(self):
        maxGain= bd.max(bd.array([self.gainR, self.gainG, self.gainB]))
        self.normGainR = self.gainR / maxGain
        self.normGainG = self.gainG / maxGain
        self.normGainB = self.gainB / maxGain


    def _FastGaussian(self, colors, perChannelSample=1):
        """
        Fast emission sampler:
          - Ignores alpha (no skew); uses standard Gaussian N(mu, sigma^2)
          - Defaults perChannelSample=1; if >1, will sample that many but still stays Gaussian-only
          - Keeps gain-based pruning via _normGain*
        """

        m = colors.shape[0]

        # Normalize colors row-wise into [0,1]
        row_max = bd.max(colors, axis=1, keepdims=True)
        denom = bd.maximum(row_max, ONE)
        colors_n = bd.clip(colors / denom, ZERO, ONE)

        muR, muG, muB = LambdaLines[self.lineR], LambdaLines[self.lineG], LambdaLines[self.lineB]

        # Keep probability includes normalized gain
        pR = bd.clip(colors_n[:, 0] * bd.array(self.normGainR), ZERO, ONE)
        pG = bd.clip(colors_n[:, 1] * bd.array(self.normGainG), ZERO, ONE)
        pB = bd.clip(colors_n[:, 2] * bd.array(self.normGainB), ZERO, ONE)

        # ------------------------------------------------------------------
        # perChannelSample == 1 fast path (common)
        # ------------------------------------------------------------------
        if perChannelSample == 1:
            lamR = bd.array(muR) + bd.array(self.sigmaR) * RNG.randn(m)
            lamG = bd.array(muG) + bd.array(self.sigmaG) * RNG.randn(m)
            lamB = bd.array(muB) + bd.array(self.sigmaB) * RNG.randn(m)

            lamR = bd.clip(lamR, 380.0, 780.0)
            lamG = bd.clip(lamG, 380.0, 780.0)
            lamB = bd.clip(lamB, 380.0, 780.0)

            keepR = RNG.rand(m) < pR
            keepG = RNG.rand(m) < pG
            keepB = RNG.rand(m) < pB

            selR = lamR[keepR]
            selG = lamG[keepG]
            selB = lamB[keepB]

            def pack1d(lams_1d, ch_idx):
                if lams_1d.size == 0:
                    return None
                ch = bd.full((lams_1d.shape[0],), bd.array(ch_idx), dtype=lams_1d.dtype)
                return bd.stack([lams_1d, ch], axis=1)

            arrR = pack1d(selR, 0.0)
            arrG = pack1d(selG, 1.0)
            arrB = pack1d(selB, 2.0)

            parts = [a for a in (arrR, arrG, arrB) if a is not None]
            if len(parts) == 0:
                return bd.zeros((0, 2), dtype=bd.float64)

            return bd.concatenate(parts, axis=0)

        # ------------------------------------------------------------------
        # Optional: perChannelSample > 1 (still Gaussian-only)
        # ------------------------------------------------------------------
        lamR = bd.array(muR) + bd.array(self.sigmaR) * RNG.randn(m, perChannelSample)
        lamG = bd.array(muG) + bd.array(self.sigmaG) * RNG.randn(m, perChannelSample)
        lamB = bd.array(muB) + bd.array(self.sigmaB) * RNG.randn(m, perChannelSample)

        lamR = bd.clip(lamR, 380.0, 780.0)
        lamG = bd.clip(lamG, 380.0, 780.0)
        lamB = bd.clip(lamB, 380.0, 780.0)

        keepR = RNG.rand(m, perChannelSample) < pR[:, bd.newaxis]
        keepG = RNG.rand(m, perChannelSample) < pG[:, bd.newaxis]
        keepB = RNG.rand(m, perChannelSample) < pB[:, bd.newaxis]

        selR = lamR[keepR]
        selG = lamG[keepG]
        selB = lamB[keepB]

        def pack(lams_1d, ch_idx):
            if lams_1d.size == 0:
                return None
            ch = bd.full((lams_1d.shape[0],), bd.array(ch_idx), dtype=lams_1d.dtype)
            return bd.stack([lams_1d, ch], axis=1)

        arrR = pack(selR, 0.0)
        arrG = pack(selG, 1.0)
        arrB = pack(selB, 2.0)

        parts = [a for a in (arrR, arrG, arrB) if a is not None]
        if len(parts) == 0:
            return bd.zeros((0, 2), dtype=bd.float64)

        return bd.concatenate(parts, axis=0)


# ==================================================================
""" ======================= End of class ======================= """
# ==================================================================


def PlotWavelengthHistogram(wavelength_ch, bin_nm=10.0, lam_min=380.0, lam_max=780.0):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    Plot a histogram over the visible spectrum.
    Input format matches NewWavelengthTest output:
        wavelength_ch: array of shape (k, 2)
          - col 0: wavelength in nm
          - col 1: channel index (0=R, 1=G, 2=B)

    Each spectral bin is shown as a single bar with stacked RGB segments
    (height = counts in that bin; segment colors follow index code 0/1/2 -> R/G/B).
    """

    # --- bring to CPU numpy (works for numpy / cupy arrays) ---
    lam = wavelength_ch[:, 0]
    ch  = wavelength_ch[:, 1].astype(np.int32)

    if hasattr(lam, "get"):  # CuPy -> NumPy
        lam = lam.get()
    if hasattr(ch, "get"):
        ch = ch.get()

    lam = np.asarray(lam, dtype=np.float64)
    ch  = np.asarray(ch, dtype=np.int32)

    # --- binning ---
    edges = np.arange(lam_min, lam_max + bin_nm, bin_nm, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (edges[1] - edges[0]) * 0.95

    # Counts per channel per bin
    counts = np.zeros((3, len(edges) - 1), dtype=np.int64)
    for c in (0, 1, 2):
        mask = (ch == c)
        counts[c], _ = np.histogram(lam[mask], bins=edges)

    # --- plot (stacked bars) ---
    fig, ax = plt.subplots()

    bottom = np.zeros_like(counts[0], dtype=np.int64)

    # index code: 0->R, 1->G, 2->B
    colors = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}
    labels = {0: "R (0)", 1: "G (1)", 2: "B (2)"}

    for c in (0, 1, 2):
        ax.bar(
            centers,
            counts[c],
            width=width,
            bottom=bottom,
            color=colors[c],
            align="center",
            label=labels[c],
            linewidth=0
        )
        bottom += counts[c]

    ax.set_xlim(lam_min, lam_max)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Count")
    ax.set_title(f"Wavelength distribution (bin = {bin_nm} nm)")
    ax.legend()

    plt.show()
    return fig, ax