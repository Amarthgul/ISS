
"""
Probability Density Function for color-wavelength conversion.

"""


from Util.Backend import backend as bd
from Util.Globals import RNG, RefreshRNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines, Channels
from Util.MathFunctions import Erf, SkewNormPDF


LAM_MIN = 380.0
LAM_MAX = 780.0


def sample_skew_normal(mu, sigma, alpha, shape):
    # Sampling helper method

    # Fast path: plain Gaussian
    if alpha == 0:
        lam = bd.array(mu) + bd.array(sigma) * RNG.randn(*shape)
        return bd.clip(lam, LAM_MIN, LAM_MAX)

    # Azzalini skew-normal sampling:
    # mu + sigma*(delta*|z0| + sqrt(1-delta^2)*z1)
    z0 = RNG.randn(*shape)
    z1 = RNG.randn(*shape)

    a = bd.array(alpha)
    delta = a / bd.sqrt(ONE + a * a)

    w = delta * bd.abs(z0) + bd.sqrt(ONE - delta * delta) * z1
    lam = bd.array(mu) + bd.array(sigma) * w
    return bd.clip(lam, LAM_MIN, LAM_MAX)



class ColorPDF:

    def __init__(self):
        # These Fraunhofer lines are also the mu of the Gaussian distribution
        self.lineR = "r"
        self.lineG = "e"
        self.lineB = "h"

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

        self.fastGaussian = False

        # Floor-lift control for the channel PDFs:
        # lifted = 1 - (1 - normalized_raw) ** floorLiftPower.
        # 1.0 preserves the original PDF; values > 1.0 flatten peaks and enrich the tails.
        self.floorLiftPower = 2
        self.floorLiftSamples = 2048

        self.normGainR = 1
        self.normGainG = 1
        self.normGainB = 1

        # Gaussian distribution at the given wavelengths tend to reduce the max value to around 0.13-0.2, which may disrupt the existing radiance expectations, so a global scalar multiple is applied for spectral response.
        self._unitScalar = 60.

        self._updateKey = None

        self._Update()


    def ColorToWavelength(self, colors, perChannelSample=4):
        """
        Convert many colors into a batch of wavelengths.

        :param colors: array of RGB colors in shape (m, 3), values in range [0, inf]
        :param perChannelSample: number of samples per color channel.

        :return: array of wavelengths in shape (k, 2), with:
                 col 0 = wavelength (nm)
                 col 1 = channel index (0=R, 1=G, 2=B)
        """
        self._Update()

        if self.fastGaussian:
            return self._FastGaussian(colors)

        m = colors.shape[0]

        # 1) Normalize colors row-wise into [0,1]
        row_max = bd.max(colors, axis=1, keepdims=True)
        denom = bd.maximum(row_max, ONE)
        colors_n = bd.clip(colors / denom, ZERO, ONE)

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
        lamR = self.SampleChannel(Channels.R, (m, perChannelSample))
        lamG = self.SampleChannel(Channels.G, (m, perChannelSample))
        lamB = self.SampleChannel(Channels.B, (m, perChannelSample))

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


    def SampleChannel(self, channel, shape):
        """
        Sample wavelengths for one RGB channel using this PDF instance's current settings.

        When fastGaussian is enabled, skew is ignored and the channel uses the standard
        Gaussian counterpart, matching ColorToWavelength's fast path.
        """
        self._Update()

        channel = int(getattr(channel, "value", channel))

        if channel == Channels.R.value:
            line, sigma, alpha, cdf, fastCdf = self.lineR, self.sigmaR, self.alphaR, self._cdfR, self._cdfFastR
        elif channel == Channels.G.value:
            line, sigma, alpha, cdf, fastCdf = self.lineG, self.sigmaG, self.alphaG, self._cdfG, self._cdfFastG
        else:
            line, sigma, alpha, cdf, fastCdf = self.lineB, self.sigmaB, self.alphaB, self._cdfB, self._cdfFastB

        if self.fastGaussian:
            if self.floorLiftPower <= 1.0:
                return sample_skew_normal(LambdaLines[line], sigma, 0, shape)

            return self._SampleFromCDF(fastCdf, shape)

        if self.floorLiftPower <= 1.0:
            return sample_skew_normal(LambdaLines[line], sigma, alpha, shape)

        return self._SampleFromCDF(cdf, shape)


    def SpectralResponse(self, wavelength, channel):
        """
        Return scalars representing spectral responses intensity of a given wavelength assigned to a given channel.

        :param channel: an array of channel indices, size (m).
        :param wavelength: an array of wavelength, size (m).

        :return: an array of spectral responses, size (m).
        """

        self._Update()

        # Output
        resp = bd.zeros_like(wavelength, dtype=bd.float64)

        # Channel masks
        mR = (channel == 0)
        mG = (channel == 1)
        mB = (channel == 2)

        # Evaluate per-channel PDF and apply *non-normalized* gains
        if bd.any(mR):
            resp[mR] = self._LiftedPDF(
                wavelength[mR], self.lineR, self.sigmaR, self.alphaR, self._pdfPeakR, self._pdfNormR
            ) * bd.array(self.gainR)

        if bd.any(mG):
            resp[mG] = self._LiftedPDF(
                wavelength[mG], self.lineG, self.sigmaG, self.alphaG, self._pdfPeakG, self._pdfNormG
            ) * bd.array(self.gainG)

        if bd.any(mB):
            resp[mB] = self._LiftedPDF(
                wavelength[mB], self.lineB, self.sigmaB, self.alphaB, self._pdfPeakB, self._pdfNormB
            ) * bd.array(self.gainB)

        return resp * self._unitScalar


    def PlotDistribution(self):
        # Draw the three skewed gaussian distribution using the line, the sigma, and alpha
        # Color them in RGB respectively

        import numpy as np
        import matplotlib.pyplot as plt
        from Util.Backend import backend_name

        self._Update()

        # Visible-ish range
        x = bd.linspace(LAM_MIN, LAM_MAX, 2000)

        muR = LambdaLines[self.lineR]
        muG = LambdaLines[self.lineG]
        muB = LambdaLines[self.lineB]

        sigR = self.sigmaR
        sigG = self.sigmaG
        sigB = self.sigmaB

        if self.fastGaussian:
            aR = aG = aB = 0
            modeLabel = "fast Gaussian"

            if self.floorLiftPower <= 1.0:
                peakR = peakG = peakB = ONE
                normR = normG = normB = ONE
            else:
                peakR, normR, _ = self._BuildPDFCache(self.lineR, sigR, aR)
                peakG, normG, _ = self._BuildPDFCache(self.lineG, sigG, aG)
                peakB, normB, _ = self._BuildPDFCache(self.lineB, sigB, aB)
        else:
            aR = self.alphaR
            aG = self.alphaG
            aB = self.alphaB
            modeLabel = "skew-normal"

            peakR, normR = self._pdfPeakR, self._pdfNormR
            peakG, normG = self._pdfPeakG, self._pdfNormG
            peakB, normB = self._pdfPeakB, self._pdfNormB

        yR = self._LiftedPDF(x, self.lineR, sigR, aR, peakR, normR) * self.normGainR
        yG = self._LiftedPDF(x, self.lineG, sigG, aG, peakG, normG) * self.normGainG
        yB = self._LiftedPDF(x, self.lineB, sigB, aB, peakB, normB) * self.normGainB
        ySum = yR + yG + yB

        # Plt, again, does not accept cupy
        if backend_name == "cupy":
            x = x.get()
            yR = yR.get()
            yG = yG.get()
            yB = yB.get()
            ySum = ySum.get()

        fig, (ax, ax_sum) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
        )
        ax.plot(x, yR, color=(1.0, 0.0, 0.0), label=f"R: mu={muR:.2f}nm sigma={sigR:g} alpha={aR:g}")
        ax.plot(x, yG, color=(0.0, 1.0, 0.0), label=f"G: mu={muG:.2f}nm sigma={sigG:g} alpha={aG:g}")
        ax.plot(x, yB, color=(0.0, 0.0, 1.0), label=f"B: mu={muB:.2f}nm sigma={sigB:g} alpha={aB:g}")

        # Optional: mark mus
        ax.axvline(muR, color=(1.0, 0.0, 0.0), linestyle="--", linewidth=1, alpha=0.4)
        ax.axvline(muG, color=(0.0, 1.0, 0.0), linestyle="--", linewidth=1, alpha=0.4)
        ax.axvline(muB, color=(0.0, 0.0, 1.0), linestyle="--", linewidth=1, alpha=0.4)

        ax_sum.plot(x, ySum, color=(0.2, 0.2, 0.2), label="Aggregate")
        ax_sum.fill_between(x, ySum, color=(0.2, 0.2, 0.2), alpha=0.18, linewidth=0)

        ax.set_xlim(LAM_MIN, LAM_MAX)
        ax.set_ylabel("PDF")
        ax.set_title(f"Channel Wavelength PDFs ({modeLabel}, floor lift k={self.floorLiftPower:g})")
        ax.legend()

        ax_sum.set_xlim(LAM_MIN, LAM_MAX)
        ax_sum.set_xlabel("Wavelength (nm)")
        ax_sum.set_ylabel("Sum")
        ax_sum.legend()

        plt.show()
        return fig, (ax, ax_sum)


    def WavelengthVis(self, inputWavelengths):

        PlotWavelengthHistogram(inputWavelengths)


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _Update(self):
        updateKey = (
            self.lineR, self.lineG, self.lineB,
            float(self.sigmaR), float(self.sigmaG), float(self.sigmaB),
            float(self.alphaR), float(self.alphaG), float(self.alphaB),
            float(self.gainR), float(self.gainG), float(self.gainB),
            float(self.floorLiftPower), int(self.floorLiftSamples),
        )

        if updateKey == self._updateKey:
            return

        maxGain = max(float(self.gainR), float(self.gainG), float(self.gainB), NEAR_ZERO)
        self.normGainR = self.gainR / maxGain
        self.normGainG = self.gainG / maxGain
        self.normGainB = self.gainB / maxGain

        if self.floorLiftPower <= 1.0:
            self._pdfPeakR = self._pdfPeakG = self._pdfPeakB = ONE
            self._pdfNormR = self._pdfNormG = self._pdfNormB = ONE
            self._cdfR = self._cdfG = self._cdfB = None
            self._cdfFastR = self._cdfFastG = self._cdfFastB = None
            self._updateKey = updateKey
            return

        sampleCount = max(2, int(self.floorLiftSamples))
        self._sampleGrid = bd.linspace(LAM_MIN, LAM_MAX, sampleCount)

        self._pdfPeakR, self._pdfNormR, self._cdfR = self._BuildPDFCache(self.lineR, self.sigmaR, self.alphaR)
        self._pdfPeakG, self._pdfNormG, self._cdfG = self._BuildPDFCache(self.lineG, self.sigmaG, self.alphaG)
        self._pdfPeakB, self._pdfNormB, self._cdfB = self._BuildPDFCache(self.lineB, self.sigmaB, self.alphaB)

        self._cdfFastR = self._cdfR if self.alphaR == 0 else self._BuildPDFCache(self.lineR, self.sigmaR, 0)[2]
        self._cdfFastG = self._cdfG if self.alphaG == 0 else self._BuildPDFCache(self.lineG, self.sigmaG, 0)[2]
        self._cdfFastB = self._cdfB if self.alphaB == 0 else self._BuildPDFCache(self.lineB, self.sigmaB, 0)[2]

        self._updateKey = updateKey


    def _RawPDF(self, wavelength, line, sigma, alpha):
        return SkewNormPDF(wavelength, LambdaLines[line], sigma, alpha)


    def _BuildPDFCache(self, line, sigma, alpha):
        raw = self._RawPDF(self._sampleGrid, line, sigma, alpha)
        peak = bd.max(raw)

        if self.floorLiftPower <= 1.0:
            pdf = raw
            norm = ONE
        else:
            raw_n = bd.clip(raw / (peak + NEAR_ZERO), ZERO, ONE)
            pdf = ONE - (ONE - raw_n) ** bd.array(self.floorLiftPower)

            dx = (LAM_MAX - LAM_MIN) / (self._sampleGrid.shape[0] - 1)
            area = bd.sum((pdf[:-1] + pdf[1:]) * 0.5) * dx
            norm = ONE / (area + NEAR_ZERO)
            pdf = pdf * norm

        cdf = self._PDFToCDF(pdf)
        return peak, norm, cdf


    def _PDFToCDF(self, pdf):
        dx = (LAM_MAX - LAM_MIN) / (self._sampleGrid.shape[0] - 1)
        seg = (pdf[:-1] + pdf[1:]) * 0.5 * dx
        cdf = bd.concatenate([bd.zeros((1,), dtype=pdf.dtype), bd.cumsum(seg)])
        return cdf / (cdf[-1] + NEAR_ZERO)


    def _LiftedPDF(self, wavelength, line, sigma, alpha, peak, norm):
        raw = self._RawPDF(wavelength, line, sigma, alpha)

        if self.floorLiftPower <= 1.0:
            return raw

        raw_n = bd.clip(raw / (peak + NEAR_ZERO), ZERO, ONE)
        lifted = ONE - (ONE - raw_n) ** bd.array(self.floorLiftPower)
        return lifted * norm


    def _SampleFromCDF(self, cdf, shape):
        u = RNG.rand(*shape)
        idx_hi = bd.searchsorted(cdf, u, side="left")
        idx_hi = bd.clip(idx_hi, 1, self._sampleGrid.shape[0] - 1)
        idx_lo = idx_hi - 1

        cdf_lo = cdf[idx_lo]
        cdf_hi = cdf[idx_hi]
        lam_lo = self._sampleGrid[idx_lo]
        lam_hi = self._sampleGrid[idx_hi]

        t = (u - cdf_lo) / (cdf_hi - cdf_lo + NEAR_ZERO)
        return bd.clip(lam_lo + t * (lam_hi - lam_lo), LAM_MIN, LAM_MAX)




    def _FastGaussian(self, colors, perChannelSample=1):
        """
        Fast emission sampler:
          - Ignores alpha (no skew); uses standard Gaussian N(mu, sigma^2)
          - Defaults perChannelSample=1; if >1, will sample that many but still stays Gaussian-only
          - Keeps gain-based pruning via _normGain*
        """

        self._Update()

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
            if self.floorLiftPower <= 1.0:
                lamR = bd.array(muR) + bd.array(self.sigmaR) * RNG.randn(m)
                lamG = bd.array(muG) + bd.array(self.sigmaG) * RNG.randn(m)
                lamB = bd.array(muB) + bd.array(self.sigmaB) * RNG.randn(m)

                lamR = bd.clip(lamR, LAM_MIN, LAM_MAX)
                lamG = bd.clip(lamG, LAM_MIN, LAM_MAX)
                lamB = bd.clip(lamB, LAM_MIN, LAM_MAX)
            else:
                lamR = self._SampleFromCDF(self._cdfFastR, (m,))
                lamG = self._SampleFromCDF(self._cdfFastG, (m,))
                lamB = self._SampleFromCDF(self._cdfFastB, (m,))

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
        if self.floorLiftPower <= 1.0:
            lamR = bd.array(muR) + bd.array(self.sigmaR) * RNG.randn(m, perChannelSample)
            lamG = bd.array(muG) + bd.array(self.sigmaG) * RNG.randn(m, perChannelSample)
            lamB = bd.array(muB) + bd.array(self.sigmaB) * RNG.randn(m, perChannelSample)

            lamR = bd.clip(lamR, LAM_MIN, LAM_MAX)
            lamG = bd.clip(lamG, LAM_MIN, LAM_MAX)
            lamB = bd.clip(lamB, LAM_MIN, LAM_MAX)
        else:
            lamR = self._SampleFromCDF(self._cdfFastR, (m, perChannelSample))
            lamG = self._SampleFromCDF(self._cdfFastG, (m, perChannelSample))
            lamB = self._SampleFromCDF(self._cdfFastB, (m, perChannelSample))

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

    The upper panel shows RGB channel distributions as stacked bars.
    The lower panel shows the aggregate distribution summed over all channels.
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

    total_counts = counts.sum(axis=0)

    # --- plot (stacked channel bars + aggregate distribution) ---
    fig, (ax, ax_sum) = plt.subplots(
        2,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

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
    ax.set_ylabel("Count")
    ax.set_title(f"Wavelength channel distributions (bin = {bin_nm} nm)")
    ax.legend()

    ax_sum.bar(
        centers,
        total_counts,
        width=width,
        color=(0.2, 0.2, 0.2),
        align="center",
        label="Aggregate"
    )
    ax_sum.set_xlim(lam_min, lam_max)
    ax_sum.set_xlabel("Wavelength (nm)")
    ax_sum.set_ylabel("Total")
    ax_sum.legend()

    plt.show()
    return fig, (ax, ax_sum)


