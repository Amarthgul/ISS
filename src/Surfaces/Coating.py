

from Util.Backend import backend as bd
from Util.Globals import RNG, LambdaLines



class Coating:
    def __init__(self):

         self.reflectKill = 0



    def ReflectionModulate(self, reflectedRaybatch):

        return reflectedRaybatch








class NaiveCoating(Coating):

    def __init__(self):
         super().__init__()

         self.reflectKill = 0.9



    def ReflectionModulate(self, reflectedRaybatch):

        # Randomly drop some of the reflected rays based on reflectKill
        if reflectedRaybatch is None or reflectedRaybatch.value is None:
            return reflectedRaybatch

        rayCount = reflectedRaybatch.value.shape[0]
        if rayCount == 0:
            return reflectedRaybatch

        dropRate = min(max(float(self.reflectKill), 0.0), 1.0)
        keepMask = RNG.rand(rayCount) >= dropRate
        reflectedRaybatch.Mask(keepMask)

        return reflectedRaybatch



class BiasedNaiveCoating(Coating):

    def __init__(self):
         super().__init__()

         self.biasKill = {
                "i" : 0,
                "h" : 0.02,
                "g" : 0.05,
                "F'": 0.2,
                "F" : 0.4,
                "e" : 0.5,
                "d" : 0.5,
                "D" : 0.5,
                "C'": 0.5,
                "C" : 0.5,
                "r" : 0.5,
                "A'": 0.5,
                "s" : 0.5,
                "t" : 0.5
            }

         self._killCache = None


    def ReflectionModulate(self, reflectedRaybatch):
        if reflectedRaybatch is None or reflectedRaybatch.value is None:
            return reflectedRaybatch

        rayCount = reflectedRaybatch.value.shape[0]
        if rayCount == 0:
            return reflectedRaybatch

        wavelengths, killRates = self._KillCurve()
        if wavelengths.shape[0] == 0:
            return reflectedRaybatch

        rayWavelengths = reflectedRaybatch.Wavelength()
        dropRate = self._KillRateForWavelengths(rayWavelengths, wavelengths, killRates)
        keepMask = RNG.rand(rayCount) >= dropRate
        reflectedRaybatch.Mask(keepMask)

        return reflectedRaybatch


    def PlotTransmission(self):
        wavelengths, killRates = self._KillCurve()
        if wavelengths.shape[0] == 0:
            return None

        plotWavelengths = wavelengths
        plotTransmission = 1.0 - killRates
        if hasattr(plotWavelengths, "get"):
            plotWavelengths = plotWavelengths.get()
            plotTransmission = plotTransmission.get()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(plotWavelengths, plotTransmission, marker="o")

        for symbol, wavelength, killRate in self._SortedBiasItems():
            ax.annotate(
                symbol,
                (wavelength, 1.0 - killRate),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
            )

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
        plt.show()

        return ax


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _KillCurve(self):
        signature = tuple(
            sorted(
                (symbol, float(killRate))
                for symbol, killRate in self.biasKill.items()
                if symbol in LambdaLines
            )
        )

        if self._killCache is not None and self._killCache[0] == signature:
            return self._killCache[1], self._killCache[2]

        sortedItems = self._SortedBiasItems()
        wavelengths = bd.array([item[1] for item in sortedItems])
        killRates = bd.array([item[2] for item in sortedItems])

        self._killCache = (signature, wavelengths, killRates)
        return wavelengths, killRates


    def _KillRateForWavelengths(self, rayWavelengths, wavelengths, killRates):
        if wavelengths.shape[0] == 1:
            return bd.full(rayWavelengths.shape, killRates[0])

        dropRate = bd.full(rayWavelengths.shape, killRates[0])

        for i in range(1, wavelengths.shape[0]):
            leftWavelength = wavelengths[i - 1]
            rightWavelength = wavelengths[i]
            leftKill = killRates[i - 1]
            rightKill = killRates[i]

            span = bd.maximum(rightWavelength - leftWavelength, 1e-12)
            ratio = bd.clip((rayWavelengths - leftWavelength) / span, 0.0, 1.0)
            localDropRate = leftKill + (rightKill - leftKill) * ratio
            inSegment = (rayWavelengths >= leftWavelength) & (rayWavelengths <= rightWavelength)
            dropRate = bd.where(inSegment, localDropRate, dropRate)

        dropRate = bd.where(rayWavelengths >= wavelengths[-1], killRates[-1], dropRate)
        return dropRate


    def _SortedBiasItems(self):
        items = []
        for symbol, killRate in self.biasKill.items():
            if symbol not in LambdaLines:
                continue

            items.append((
                symbol,
                float(LambdaLines[symbol]),
                self._ClampKillRate(killRate),
            ))

        items.sort(key=lambda item: item[1])
        return items


    def _ClampKillRate(self, killRate):
        return min(max(float(killRate), 0.0), 1.0)

