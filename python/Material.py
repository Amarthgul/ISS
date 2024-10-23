import numpy as np
import pandas as pd 
import os
import math

import matplotlib.pyplot as plt

# Primarily using the LambdaLines definition 
from Util import LambdaLines


def Schott(x, coef):
    """
    Schott function for inverse calculating the material. 
    :param lam: lambda wavelength in nanometers. 
    """
    a0 = coef[0]
    a1 = coef[1]
    a2 = coef[2]
    a3 = coef[3]
    a4 = coef[4]
    a5 = coef[5]
    n2 = a0 + a1* x**2 + a2 * x**(-2) + a3 * x**(-4) + a4 * x**(-6) + a5 * x**(-8)
    print(n2)
    return np.sqrt(n2)


def inv_schott(lambd: np.ndarray, a: np.ndarray, powers: np.ndarray) -> np.ndarray:
    return np.sqrt(inv_schott_squared(lambd, a, powers))

def inv_schott_squared(lambd: np.ndarray, a: np.ndarray, powers: np.ndarray) -> np.ndarray:
    terms = np.power.outer(lambd, powers)
    return terms @ a


class Material:

    def __init__(self, name = "AIR"):
        self.name = name 

        self.category = None 
        self.Formula = None 

        self.glassTable = None 
        self.coef = [] 

        self.Startup()

    def RI(self, lam):
        """
        The index of refraction of the material at given wavelength. 
        : param lam: lambda wavelength. 
        """
        if(self.name == "AIR"):
            # Air got the constant RI of 1 
            return 1 
        else:
            # Non air material is sent to further inquiries 
            return self._RI(lam)

    def n_e(self):
        return self.RI(LambdaLines["e"])
    
    def n_d(self):
        return self.RI(LambdaLines["d"])
    
    def V_e(self):
        return ( self.RI(LambdaLines["e"]) - 1 ) / \
            (self.RI(LambdaLines["F'"]) - self.RI(LambdaLines["C'"]))

    def V_d(self):
        return ( self.RI(LambdaLines["d"]) - 1 ) / \
            (self.RI(LambdaLines["F"]) - self.RI(LambdaLines["C"]))

    def DrawRI(self, UV=380, IR=720):
        lam = np.arange(UV, IR, dtype=float) 
        RI = self._RI(lam)

        plt.plot(lam, RI)
        plt.show()

    def Startup(self):

        # TODO: the read table and lookup turned out to be the 
        # slowest part of the entire program, find a way to accelerate this 

        if(self.name == "AIR"):
            return 
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, "AbbeGlassTable.xlsx")
            df = pd.read_excel(file_path)
            found = df[df["Name"] == self.name].iloc[0]
            # Same name material should have the same parameter so it should not matter 
            formula = found["Formula"] 

            if (formula == "Schott"):
                self.Formula = "Schott"
                self._decodeSchott(found)
            elif (formula == "Sellmeier1"):
                self.Formula = "Sellmeier1"
                self._decodeSellmeier1(found)
            elif (formula == "Extended 2"):
                self.Formula = "Extended 2"
                self._decodeExtended_2(found)
            elif (formula == "Extended 3"):
                self.Formula = "Extended 3"
                self._decodeExtended_3(found)
    
    def Test(self, var):
        ne = Schott(0.54607, var)
        nfp = Schott(0.47999, var)
        ncp = Schott(0.64385, var)
        print("ne: ", ne)
        print("Ve: ", (  (ne-1) / (nfp-ncp) ))

    def InverseMaterial(self, n, V, useNe = True):

        # Not really working... 

        # The RI regression equation of the long wavelength was calculated externally 
        
        shorter = LambdaLines["F'"]
        short = LambdaLines["F"]
        neighbor = LambdaLines["e"]
        middle = LambdaLines["d"]
        longc = LambdaLines["C'"]
        longer = LambdaLines["C"]
        n_long = 0.984 * n + 0.0246       # n_C'

        n_shorter = ( (n-1) / V) + n_long # n_F'
        n_short = 1.03 * n -0.0418       # n_F
        n_neighbor = n                    # n_e
        n_mid = 0.986 * n + 0.0202         # n_d 
        n_long = 0.971 * n + 0.0406       # n_C' 
        n_longer = 0.968 * n + 0.0443     # n_C

        x_data = np.array([longer,      longc,   middle,     neighbor,       short,      shorter])
        y_data = np.array([n_longer,    n_long, n_mid,      n_neighbor,     n_short,    n_shorter])

        lambda_nm = np.array((longer, longc, middle, neighbor, short, shorter))
        lambda_um = lambda_nm*1e-3 # Converting to micrometer 
        n_all = np.array((n_longer, n_long, n_mid, n_neighbor, n_short, n_shorter))

        fig, ax = plt.subplots()
        lambda_hires = np.linspace(start=lambda_um.min(), stop=lambda_um.max(), num=501)
        ax.scatter(lambda_um, n_all, label='experiment')


        for lowest_power in range(0, -9, -2):
            powers = np.arange(2, lowest_power - 1, -2)
            a, residuals, rank, singular = np.linalg.lstsq(
                a=np.power.outer(lambda_um, powers),
                b=n_all**2, rcond=None,
            )
            print("a:", a, " Lowest power: ", lowest_power,  "\npowers   ", powers, "\n")
            ax.plot(lambda_hires, inv_schott(lambda_hires, a, powers), label=f'{lowest_power}th power')

        ax.legend()
        plt.show()


        # popt, pcov = curve_fit(_InvSchott, x_data, y_data, [2.75118, -0.01055, 0.02357, 0.00084, -0.00003, 0.00001])

        #print(popt, pcov)

    # ========================================================================
    """ ============================ Private ============================== """
    # ========================================================================
    def _RI(self, wavelength = 550):
        if(self.Formula == "Schott"):
            return self._Schott(wavelength)
        elif(self.Formula == "Sellmeier1"):
            return self._Sellmeier1(wavelength)
        elif(self.Formula == "Extended 2"):
            return self._Extended_2(wavelength)
        elif(self.Formula == "Extended 3"):
            return self._Extended_3(wavelength)
        
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

    def _Schott(self, lam):
        """
        Calculate refraction index based on Schott. 
        :param lam: lambda wavelength in nanometers. 
        """
        a0 = self.coef[0]
        a1 = self.coef[1]
        a2 = self.coef[2]
        a3 = self.coef[3]
        a4 = self.coef[4]
        a5 = self.coef[5]
        lam = np.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1* lam**2 + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8)
        return np.sqrt(n2)

    def _decodeSellmeier1(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="K1")[0] + 1][0],
            df[np.where(df=="L1")[0] + 1][0],
            df[np.where(df=="K2")[0] + 1][0],
            df[np.where(df=="L2")[0] + 1][0],
            df[np.where(df=="K3")[0] + 1][0],
            df[np.where(df=="L3")[0] + 1][0]
        ]

    def _Sellmeier1(self, lam):
        k1 = self.coef[0]
        l1 = self.coef[1]
        k2 = self.coef[2]
        l2 = self.coef[3]
        k3 = self.coef[4]
        l3 = self.coef[5]
        lam = np.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = (k1 * lam**2) / (lam**2 - l1) + (k2 * lam**2) / (lam**2 - l2) + (k3 * lam**2) / (lam**2 - l3) + 1
        return np.sqrt(n2) 

    def _decodeExtended_2(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="A0")[0] + 1][0],
            df[np.where(df=="A1")[0] + 1][0],
            df[np.where(df=="A2")[0] + 1][0],
            df[np.where(df=="A3")[0] + 1][0],
            df[np.where(df=="A4")[0] + 1][0],
            df[np.where(df=="A5")[0] + 1][0],
            df[np.where(df=="A6")[0] + 1][0],
            df[np.where(df=="A7")[0] + 1][0],
        ]

    def _Extended_2(self, lam):
        a0 = self.coef[0]
        a1 = self.coef[1]
        a2 = self.coef[2]
        a3 = self.coef[3]
        a4 = self.coef[4]
        a5 = self.coef[5]
        a6 = self.coef[6]
        a7 = self.coef[7]
        lam = np.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1 * lam**(2) + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8) + a6 * lam**(4) + a7 * lam**(6)
        return np.sqrt(n2)

    def _decodeExtended_3(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="A0")[0] + 1][0],
            df[np.where(df=="A1")[0] + 1][0],
            df[np.where(df=="A2")[0] + 1][0],
            df[np.where(df=="A3")[0] + 1][0],
            df[np.where(df=="A4")[0] + 1][0],
            df[np.where(df=="A5")[0] + 1][0],
            df[np.where(df=="A6")[0] + 1][0],
            df[np.where(df=="A7")[0] + 1][0],
            df[np.where(df=="A8")[0] + 1][0]
        ]

    def _Extended_3(self, lam):
        a0 = self.coef[0]
        a1 = self.coef[1]
        a2 = self.coef[2]
        a3 = self.coef[3]
        a4 = self.coef[4]
        a5 = self.coef[5]
        a6 = self.coef[6]
        a7 = self.coef[7]
        a8 = self.coef[8]
        lam = np.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1 * lam**(2) + a2 * lam**(4) + a3 * lam**(-2) + a4 * lam**(-4) + a5 * lam**(-6) + a6 * lam**(-8) + a7 * lam**(-10) + a8 * lam**(-12)
        return np.sqrt(n2)

    # TODO: add more decoder and formula here if needed 

    # ========================================================================
    """ ============================ Archive ============================== """
    # ========================================================================

def main():
    newglass = Material("E-KZFH1")
    #newglass.DrawRI()

    paras = newglass.InverseMaterial(1.7899, 48)
    #newglass.Test([-9.47548462,  6.75216293,  9.03861417, -3.13234753,  0.53095114, -0.03512969])
    #print("fit: ", paras)
    # [  5.337381, 14.407971, -2.9712481, -13.900439, -12.490178, 5.0407643]

if __name__ == "__main__":
    main() 
