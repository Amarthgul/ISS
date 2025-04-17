

import pandas as pd 
import os
import math
import numpy as np 
import numbers

import matplotlib.pyplot as plt
from Util.Backend import backend as bd
from Util.Backend import backend_name
from Util.Globals import ZERO, ONE, TWO, LambdaLines
from Util.Misc import RectPath


GlassTablePath = RectPath("resources/AbbeGlassTable.xlsx")


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
            return bd.ones_like(lam)
        else:
            # Non air material is sent to further inquiries 
            return self._RI(lam)

    def n_e(self):
        return self.RI(LambdaLines["e"])
    
    def n_d(self):
        return self.RI(LambdaLines["d"])
    
    def V_e(self):
        return ( self.RI(LambdaLines["e"]) - ONE ) / \
            (self.RI(LambdaLines["F'"]) - self.RI(LambdaLines["C'"]))

    def V_d(self):
        return ( self.RI(LambdaLines["d"]) - ONE ) / \
            (self.RI(LambdaLines["F"]) - self.RI(LambdaLines["C"]))

    def V_D(self):
        return ( self.RI(LambdaLines["D"]) - ONE ) / \
            (self.RI(LambdaLines["F"]) - self.RI(LambdaLines["C"]))



    def DrawRI(self, UV=380.0, IR=720.0):
        lam = np.arange(UV, IR)
        RI = self._RI(lam)

        if(backend_name == "cupy"):
            lam = bd.asnumpy(lam)
            RI = bd.asnumpy(RI)

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
            elif (formula == "Conrady"):
                self.Formula = "Conrady"
                self._decodeConrady(found)
            elif (formula == "Herzberger"):
                self.Formula = "Herzberger"
                self._decodeHerzberger(found)
            elif (formula == "Sellmeier1"):
                self.Formula = "Sellmeier1"
                self._decodeSellmeier1(found)
            elif (formula == "Sellmeier3"):
                self.Formula = "Sellmeier3"
                self._decodeSellmeier3(found)
            elif (formula == "Sellmeier4"):
                self.Formula = "Sellmeier4"
                self._decodeSellmeier4(found)
            elif (formula == "Sellmeier5"):
                self.Formula = "Sellmeier5"
                self._decodeSellmeier5(found)
            elif (formula == "Extended 2"):
                self.Formula = "Extended 2"
                self._decodeExtended_2(found)
            elif (formula == "Extended 3"):
                self.Formula = "Extended 3"
                self._decodeExtended_3(found)

        # print(self.Formula, ":  ", self.coef)

        # After reading the parameters, try to convert the data to cupy 
        self.coef = bd.asarray(self.coef)
    

    def Test(self, var):
        pass 
   
   
    # ========================================================================
    """ ============================ Private ============================== """
    # ========================================================================


    def _RI(self, wavelength = 550):
        ior = 0

        if(self.Formula == "Schott"):
            ior = self._Schott(wavelength)
        elif (self.Formula == "Conrady"):
            ior = self._Conrady(wavelength)
        elif (self.Formula == "Herzberger"):
            ior = self._Herzberger(wavelength)
        elif (self.Formula == "Sellmeier1"):
            ior = self._Sellmeier1(wavelength)
        elif (self.Formula == "Sellmeier3"):
            ior = self._Sellmeier3(wavelength)
        elif (self.Formula == "Sellmeier4"):
            ior = self._Sellmeier4(wavelength)
        elif (self.Formula == "Sellmeier5"):
            ior = self._Sellmeier5(wavelength)
        elif (self.Formula == "Extended 2"):
            ior = self._Extended_2(wavelength)
        elif (self.Formula == "Extended 3"):
            ior = self._Extended_3(wavelength)

        if(ior is None):
            ior = 0

        return ior


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
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = a0 + a1* lam**2 + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8)
        return bd.sqrt(n2)


    def _decodeSellmeier1(self, df):
        """
        There is one material using Sellmeier 1 whose name is literally K3, which causes problem in coefficient lookup, thus born this complicated mess.

        """

        # Convert to a plain Python list for fast iteration
        row_vals = df.to_list()

        # ------------------------------------------------------------------
        # 1) Build a token → value dictionary by scanning pair‑wise
        # ------------------------------------------------------------------
        token_value = {}
        i = 0
        while i < len(row_vals) - 1:
            token = row_vals[i]
            value = row_vals[i + 1]

            # Accept the pair only if the token is a str and the value is numeric
            if isinstance(token, str) and isinstance(value, numbers.Number):
                # Only record the *first* numeric value we encounter for each token
                token_value.setdefault(token, value)
            i += 2  # jump to the next candidate pair

        # ------------------------------------------------------------------
        # 2) Pull the coefficients we need from the dictionary
        # ------------------------------------------------------------------
        try:
            self.coef = [
                token_value["K1"],
                token_value["L1"],
                token_value["K2"],
                token_value["L2"],
                token_value["K3"],
                token_value["L3"],
            ]
        except KeyError as missing:
            raise ValueError(
                f"Coefficient {missing} not found for material {self.name}. "
                "Check the source table formatting."
            )

    def _Sellmeier1(self, lam):
        k1 = self.coef[0]
        l1 = self.coef[1]
        k2 = self.coef[2]
        l2 = self.coef[3]
        k3 = self.coef[4]
        l3 = self.coef[5]
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = (k1 * lam**2) / (lam**2 - l1) + (k2 * lam**2) / (lam**2 - l2) + (k3 * lam**2) / (lam**2 - l3) + 1
        return bd.sqrt(n2) 


    def _decodeSellmeier3(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="K1")[0] + 1][0],
            df[np.where(df=="L1")[0] + 1][0],
            df[np.where(df=="K2")[0] + 1][0],
            df[np.where(df=="L2")[0] + 1][0],
            df[np.where(df=="K3")[0] + 1][0],
            df[np.where(df=="L3")[0] + 1][0],
            df[np.where(df=="K4")[0] + 1][0],
            df[np.where(df=="L4")[0] + 1][0]
        ]

    def _Sellmeier3(self, lam):
        k1 = self.coef[0]
        l1 = self.coef[1]
        k2 = self.coef[2]
        l2 = self.coef[3]
        k3 = self.coef[4]
        l3 = self.coef[5]
        k4 = self.coef[4]
        l4 = self.coef[5]
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = (k1 * lam**2) / (lam**2 - l1) + (k2 * lam**2) / (lam**2 - l2) + (k3 * lam**2) / (lam**2 - l3) + (k4 * lam**2) / (lam**2 - l4) + 1
        return bd.sqrt(n2)


    def _decodeSellmeier4(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="A")[0] + 1][0],
            df[np.where(df=="B")[0] + 1][0],
            df[np.where(df=="C")[0] + 1][0],
            df[np.where(df=="D")[0] + 1][0],
            df[np.where(df=="E")[0] + 1][0],
        ]

    def _Sellmeier4(self, lam):
        A = self.coef[0]
        B = self.coef[1]
        C = self.coef[2]
        D = self.coef[3]
        E = self.coef[4]

        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = A + (B * lam**2)/(lam**2 - C) + (D * lam**2)/(lam**2 - E)
        return bd.sqrt(n2)


    def _decodeSellmeier5(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="K1")[0] + 1][0],
            df[np.where(df=="L1")[0] + 1][0],
            df[np.where(df=="K2")[0] + 1][0],
            df[np.where(df=="L2")[0] + 1][0],
            df[np.where(df=="K3")[0] + 1][0],
            df[np.where(df=="L3")[0] + 1][0],
            df[np.where(df=="K4")[0] + 1][0],
            df[np.where(df=="L4")[0] + 1][0],
            df[np.where(df=="K5")[0] + 1][0],
            df[np.where(df=="L5")[0] + 1][0]
        ]

    def _Sellmeier5(self, lam):
        k1 = self.coef[0]
        l1 = self.coef[1]
        k2 = self.coef[2]
        l2 = self.coef[3]
        k3 = self.coef[4]
        l3 = self.coef[5]
        k4 = self.coef[4]
        l4 = self.coef[5]
        k5 = self.coef[4]
        l5 = self.coef[5]
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = (k1 * lam**2) / (lam**2 - l1) + (k2 * lam**2) / (lam**2 - l2) + (k3 * lam**2) / (lam**2 - l3) + (k4 * lam**2) / (lam**2 - l4) + (k5 * lam**2) / (lam**2 - l5) + 1
        return bd.sqrt(n2)


    def _decodeConrady(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df == "N0")[0] + 1][0],
            df[np.where(df == "A")[0] + 1][0],
            df[np.where(df == "B")[0] + 1][0],
        ]

    def _Conrady(self, lam):
        N0 = self.coef[0]
        A = self.coef[1]
        B = self.coef[2]

        lam = self._LamUnitConversion(lam)  # Convert to micrometers to use in the formula
        n = N0 + A/lam + B/(lam**3.5)
        return N0 + A/lam + B/(lam**3.5)


    def _decodeHerzberger(self, df):
        df = df.to_numpy()
        self.coef = [
            df[np.where(df=="A")[0] + 1][0],
            df[np.where(df=="B")[0] + 1][0],
            df[np.where(df=="C")[0] + 1][0],
            df[np.where(df=="D")[0] + 1][0],
            df[np.where(df=="E")[0] + 1][0],
            df[np.where(df=="F")[0] + 1][0],
        ]

    def _Herzberger(self, lam):
        A = self.coef[0]
        B = self.coef[1]
        C = self.coef[2]
        D = self.coef[3]
        E = self.coef[4]
        F = self.coef[5]

        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        L = 1 / (lam**2 - 0.028)

        return A + B*L + C* L**2 + D* lam**2 + E* lam**4 + F* lam**6


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
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = a0 + a1 * lam**(2) + a2 * lam**(-2) + a3 * lam**(-4) + a4 * lam**(-6) + a5 * lam**(-8) + a6 * lam**(4) + a7 * lam**(6)
        return bd.sqrt(n2)


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
        lam = self._LamUnitConversion(lam) # Convert to micrometers to use in the formula
        n2 = a0 + a1 * lam**(2) + a2 * lam**(4) + a3 * lam**(-2) + a4 * lam**(-4) + a5 * lam**(-6) + a6 * lam**(-8) + a7 * lam**(-10) + a8 * lam**(-12)
        return bd.sqrt(n2)


    def _LamUnitConversion(self, lam):
        if isinstance(lam, float):
            return lam/1000.0
        else:
            return bd.array(bd.copy(lam) / 1000.0)


class MonochromaticMaterial(Material):
    def __init__(self, RI=1.5, name="MONO"):
        super().__init__(name)
        self.monoRI = RI


    def RI(self, lam):
        """
        For mono material, the RI is a constant regardless of wavelength. 
        """
        return bd.ones_like(lam) * self.monoRI


def CreateMaterialRefractiveIndicesTable(
        excel_in=RectPath("resources/AbbeGlassTable.xlsx"),
        excel_out=RectPath("resources/MaterialRefractionIndices.xlsx")
):
    """
    Reads all materials from 'AbbeGlassTable.xlsx', computes RI at each
    of the defined wavelength lines, and writes them to a new Excel file.

    This is extremely computationally ineffective, but man who cares, you only need to run it once whenever the AbbeGlassTable is updated.
    """

    # 1. Read the glass table. If 'Material.py' is configured to pre-read
    #    the file, you can alternatively refer to `Material.GlassTable`.
    glass_df = pd.read_excel(excel_in)

    global LambdaLines
    global GlassTable

    # Unique combinations of Catalogue + Name
    unique_pairs = glass_df[["Cate", "Name"]].drop_duplicates()

    results = []

    # ------------------------------------------------------------------
    # 2) iterate over every unique (Cate, Name) pair
    # ------------------------------------------------------------------
    original_table = GlassTable  # keep a copy to restore later

    for _, pair in unique_pairs.iterrows():
        cate_val = pair["Cate"]
        name_val = pair["Name"]

        # ------------------------------------------------------------------
        # 2a) isolate the single row matching BOTH Cate and Name
        # ------------------------------------------------------------------
        filtered_table = glass_df[(glass_df["Cate"] == cate_val) &
                                  (glass_df["Name"] == name_val)]

        # Replace the global GlassTable just for the upcoming Material() call
        GlassTable = filtered_table

        # Instantiate the material – it now sees only the correct row
        mat_obj = Material(name_val)

        # Restore the full table so the next iteration starts clean
        GlassTable = original_table

        # ------------------------------------------------------------------
        # 2b) collect output for this pair
        # ------------------------------------------------------------------
        row_data = {"Cate": cate_val, "Name": name_val}

        for line_lbl, λ_nm in LambdaLines.items():
            ri = float(mat_obj.RI(λ_nm))  # scalar – ensure python float
            row_data[line_lbl] = ri

        results.append(row_data)

    # ------------------------------------------------------------------
    # 3) write everything out
    # ------------------------------------------------------------------
    out_df = pd.DataFrame(results)
    out_df.to_excel(excel_out, index=False)
    print(f"Refractive indices saved to: {excel_out}")


def AppendAbbeNumbers(
    excel_in=RectPath("resources/MaterialRefractionIndices.xlsx"),
    excel_out=RectPath("resources/MaterialRefractionIndices.xlsx")
):
    """
    After calling CreateMaterialRefractiveIndicesTable, call this to create the corresponding V_d, V_D, and V_e.
    """

    # 1. Read the existing table of refractive indices
    df = pd.read_excel(excel_in)

    # 2. Prepare lists for Abbe numbers
    Vd_list = []
    VD_list = []
    Ve_list = []

    # 3. We will re-instantiate each Material by 'Name' in df
    #    Then compute V_d, V_D, and V_e
    for _, row in df.iterrows():
        mat_name = row["Name"]

        mat_obj = Material(mat_name)  # Re-instantiate with that name

        # Compute Abbe numbers (convert array-like to float)
        Vd_val = float(mat_obj.V_d())  # uses F, C lines
        VD_val = float(mat_obj.V_D())  # uses F, C lines but D line for numerator
        Ve_val = float(mat_obj.V_e())  # uses F', C' lines but e line for numerator

        Vd_list.append(Vd_val)
        VD_list.append(VD_val)
        Ve_list.append(Ve_val)

    # 4. Insert these columns into the DataFrame
    df["V_d"] = Vd_list
    df["V_D"] = VD_list
    df["V_e"] = Ve_list

    # 5. Save to a new Excel file
    df.to_excel(excel_out, index=False)
    print(f"Abbe numbers appended and saved to: {excel_out}")



def main():
    # newglass = Material("E-KZFH1")
    # newglass = Material("KRS5")
    # newglass.DrawRI()
    # print(newglass.coef)

    # CreateMaterialRefractiveIndicesTable()
    AppendAbbeNumbers()


 
if __name__ == "__main__":
    main() 
