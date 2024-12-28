

import pandas as pd 
import os
import math

import matplotlib.pyplot as plt
from Util.Backend import backend as bd

# Primarily using the LambdaLines definition 
from Util.Misc import LambdaLines

# Load the material sheet globally to avoid repeatly open-close 
GlassTablePath = r"resources/AbbeGlassTable.xlsx"

PreRead = True 
GlassTable = None 
if(PreRead):
    GlassTable = pd.read_excel(GlassTablePath)

def MaterialClear():
    del GlassTable

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
        lam = bd.arange(UV, IR, dtype=float) 
        RI = self._RI(lam)

        plt.plot(lam, RI)
        plt.show()

    def Startup(self):

        # TODO: the read table and lookup turned out to be the 
        # slowest part of the entire program, find a way to accelerate this 

        if(self.name == "AIR"):
            return 
        else:
            if(PreRead):
                # Pre read to save open-close time 
                df = GlassTable
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
        pass 
   
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
            df[bd.where(df=="A0")[0] + 1][0],
            df[bd.where(df=="A1")[0] + 1][0],
            df[bd.where(df=="A2")[0] + 1][0],
            df[bd.where(df=="A3")[0] + 1][0],
            df[bd.where(df=="A4")[0] + 1][0],
            df[bd.where(df=="A5")[0] + 1][0]
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
        lam = bd.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1* lam**2 + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8)
        return bd.sqrt(n2)

    def _decodeSellmeier1(self, df):
        df = df.to_numpy()
        self.coef = [
            df[bd.where(df=="K1")[0] + 1][0],
            df[bd.where(df=="L1")[0] + 1][0],
            df[bd.where(df=="K2")[0] + 1][0],
            df[bd.where(df=="L2")[0] + 1][0],
            df[bd.where(df=="K3")[0] + 1][0],
            df[bd.where(df=="L3")[0] + 1][0]
        ]

    def _Sellmeier1(self, lam):
        k1 = self.coef[0]
        l1 = self.coef[1]
        k2 = self.coef[2]
        l2 = self.coef[3]
        k3 = self.coef[4]
        l3 = self.coef[5]
        lam = bd.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = (k1 * lam**2) / (lam**2 - l1) + (k2 * lam**2) / (lam**2 - l2) + (k3 * lam**2) / (lam**2 - l3) + 1
        return bd.sqrt(n2) 

    def _decodeExtended_2(self, df):
        df = df.to_numpy()
        self.coef = [
            df[bd.where(df=="A0")[0] + 1][0],
            df[bd.where(df=="A1")[0] + 1][0],
            df[bd.where(df=="A2")[0] + 1][0],
            df[bd.where(df=="A3")[0] + 1][0],
            df[bd.where(df=="A4")[0] + 1][0],
            df[bd.where(df=="A5")[0] + 1][0],
            df[bd.where(df=="A6")[0] + 1][0],
            df[bd.where(df=="A7")[0] + 1][0],
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
        lam = bd.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1 * lam**(2) + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8) + a6 * lam**(4) + a7 * lam**(6)
        return bd.sqrt(n2)

    def _decodeExtended_3(self, df):
        df = df.to_numpy()
        self.coef = [
            df[bd.where(df=="A0")[0] + 1][0],
            df[bd.where(df=="A1")[0] + 1][0],
            df[bd.where(df=="A2")[0] + 1][0],
            df[bd.where(df=="A3")[0] + 1][0],
            df[bd.where(df=="A4")[0] + 1][0],
            df[bd.where(df=="A5")[0] + 1][0],
            df[bd.where(df=="A6")[0] + 1][0],
            df[bd.where(df=="A7")[0] + 1][0],
            df[bd.where(df=="A8")[0] + 1][0]
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
        lam = bd.copy(lam) / 1000.0 # Convert to micrometers to use in the formula 
        n2 = a0 + a1 * lam**(2) + a2 * lam**(4) + a3 * lam**(-2) + a4 * lam**(-4) + a5 * lam**(-6) + a6 * lam**(-8) + a7 * lam**(-10) + a8 * lam**(-12)
        return bd.sqrt(n2)

    # TODO: add more decoder and formula here if needed 

    # ========================================================================
    """ ============================ Archive ============================== """
    # ========================================================================

def main():
    newglass = Material("E-KZFH1")
    newglass.DrawRI()

 
if __name__ == "__main__":
    main() 
