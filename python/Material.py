import numpy as np
import pandas as pd 
import os

import matplotlib.pyplot as plt

class Material:

    def __init__(self, name):
        self.name = name 

        self.category = None 
        self.Formula = None 

        self.glassTable = None 
        self.coef = [] 

        self.Startup()

    def RI(self, lam):
        if(self.name == "AIR"):
            return 1 
        else:
            pass 

    def DrawRI(self, UV=380, IR=720):
        lam = np.arange(UV, IR, dtype=float) / 1000.0
        RI = []

        if(self.Formula == "Schott"):
            RI = self._Schott(lam, self.coef[0], self.coef[1], self.coef[2], self.coef[3],self.coef[4], self.coef[5])

        plt.plot(lam, RI)
        plt.show()

    def Startup(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "AbbeGlassTable.xlsx")
        df = pd.read_excel(file_path)
        found = df[df["Name"] == self.name].iloc[0]
        # Same name material should have the same parameter so it should not matter 
        if (found["Formula"] == "Schott"):
            self.Formula = "Schott"
            self._decodeSchott(found)

    def RI(self, wavelength = 550):
        if(self.Formula == "Schott"):
            return self._Schott(wavelength)

    # ============================================================================
    """ ============================== Private ================================ """
    # ============================================================================

    def _decodeSchott(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="A0")[0] + 1][0],
            df[np.where(df=="A1")[0] + 1][0],
            df[np.where(df=="A2")[0] + 1][0],
            df[np.where(df=="A3")[0] + 1][0],
            df[np.where(df=="A4")[0] + 1][0],
            df[np.where(df=="A5")[0] + 1][0]
        ]
        print("Coefs  ", self.coef)

    def _Schott(self, lam, a0, a1, a2, a3, a4, a5):
        n2 = a0 + a1* lam**2 + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8)
        print(n2)
        return np.sqrt(n2)




def main():
    newglass = Material("BAF1")
    newglass.DrawRI()

if __name__ == "__main__":
    main() 
