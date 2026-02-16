
"""
Probability Density Function for color-wavelength conversion.

"""


from Util.Backend import backend as bd
from Util.Globals import RNG, RefreshRNG, NEAR_ZERO, AXIAL_ZERO, ZERO, ONE, TWO, LambdaLines


class ColorPDF:

    def __init__(self):
        # These Fraunhofer lines are also the mu of the Gaussian distribution
        self.lineR = "C'"
        self.lineG = "e"
        self.lineB = "g"

        # Sigma parameter for the three Gaussian respectively
        self.sigmaR = 30
        self.sigmaG = 20
        self.sigmaB = 20

        # Alpha set is for skewed Gaussian, when set to 0 they're just standard Gaussian
        self.alphaR = 0
        self.alphaG = 0
        self.alphaB = 0


    def ColorToWavelength(self, colors, perChannelSample=2):
        """
        Convert many colors into a batch of wavelengths.

        :param colors: array of RGB colors in shape (m, 3), values in range [0, inf]
        :param perChannelSample: number of samples per color channel.

        :return: array of wavelengths in shape (k, 2), with:
                 col 0 = wavelength (nm)
                 col 1 = channel index (0=R, 1=G, 2=B)
        """

        m = colors.shape[0]

        # 1) Normalize colors row-wise into [0,1]
        # (preserve ratios; if max=0, leave as 0)
        row_max = bd.max(colors, axis=1, keepdims=True)
        denom = bd.maximum(row_max, ONE)
        colors_n = bd.clip(colors / denom, ZERO, ONE)

        # 2) Sample (skew-)Gaussian per channel for each color: shape (m, perChannelSample)
        # Azzalini skew-normal: mu + sigma*(delta*|z0| + sqrt(1-delta^2)*z1)
        def sample_skew_normal(mu, sigma, alpha):
            z0 = RNG.randn(m, perChannelSample)
            z1 = RNG.randn(m, perChannelSample)

            # alpha is scalar; make it backend-friendly
            a = bd.array(alpha)
            delta = a / bd.sqrt(ONE + a * a)

            w = delta * bd.abs(z0) + bd.sqrt(ONE - delta * delta) * z1
            lam = bd.array(mu) + bd.array(sigma) * w

            # Clamp to a sane visible-ish range (optional but helps tame tails)
            return bd.clip(lam, 380.0, 780.0)

        muR, muG, muB = LambdaLines[self.lineR], LambdaLines[self.lineG], LambdaLines[self.lineB]
        lamR = sample_skew_normal(muR, self.sigmaR, self.alphaR)
        lamG = sample_skew_normal(muG, self.sigmaG, self.alphaG)
        lamB = sample_skew_normal(muB, self.sigmaB, self.alphaB)

        # 3) Randomly cull wavelengths per channel according to normalized RGB
        keepR = RNG.rand(m, perChannelSample) < colors_n[:, 0:1]
        keepG = RNG.rand(m, perChannelSample) < colors_n[:, 1:2]
        keepB = RNG.rand(m, perChannelSample) < colors_n[:, 2:3]

        selR = lamR[keepR]
        selG = lamG[keepG]
        selB = lamB[keepB]

        # 4) Append channel index column, producing (k,2) blocks
        def pack(lams_1d, ch_idx):
            # lams_1d is 1D after masking
            if lams_1d.size == 0:
                return None
            ch = bd.full((lams_1d.shape[0],), bd.array(ch_idx), dtype=lams_1d.dtype)
            return bd.stack([lams_1d, ch], axis=1)

        arrR = pack(selR, 0.0)
        arrG = pack(selG, 1.0)
        arrB = pack(selB, 2.0)

        # 5) Stack together
        parts = [a for a in (arrR, arrG, arrB) if a is not None]
        if len(parts) == 0:
            return bd.zeros((0, 2), dtype=bd.float64)

        return bd.concatenate(parts, axis=0)


    def WavelengthAggregation(self, anotherPDF):

        pass


    def WavelengthVis(self, inputWavelengths):

        PlotWavelengthHistogram(inputWavelengths)


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