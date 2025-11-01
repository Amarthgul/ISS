# Parse a Zemax .zmx (plain-text) file into Python structures
# and extract per-surface fields depending on TYPE (STANDARD vs EVENASPH).
# The code is written to be robust to spacing and extra fields.
#
# It will also run on the uploaded file and show a quick preview.

import re
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List, Any, Tuple
import shlex


from Surfaces.Surface import Surface
from Surfaces.EvenAspheric import EvenAspheric
from Surfaces.Stop import Stop
from Lens import Lens
from Util.Globals import INFINITY, ZERO, DEFAULT_MAT_NAME


class LensFromZmx:
    def __init__(self, path):
        self.path = path
        self.content = None

        self.lens = None

        self._SurfDict = None # This is a list of dict of all surfaces parsed form the zmx file

        self.ParseFile()


    def ParseFile(self):
        self.ReadFile()
        self.SpiltSurface()
        #self._PrintDictList(self._SurfDict)
        self.ConvertIntoLens()


    def GetLens(self):
        return self.lens


    def ReadFile(self):
        with open(self.path, 'r', encoding='utf-16', errors='ignore') as f:
            self.content = f.read()
        return self.content


    def SpiltSurface(self):
        """
        Parses a Zemax .zmx file into a list of per-surface dictionaries.

        - Repeated keys are collected as lists (list of token lists).
        - Keys that appear only once are flattened to a single token list.
        - PARM lines are additionally:
            * collected into PARM = [[index:int, value:float], ...]
            * exposed as individual keys PARM1, PARM2, ... each a single float
        """

        blocks = re.split(r'(?=^SURF\s)', self.content, flags=re.MULTILINE)
        surfaces = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.splitlines()
            surface_data = {}
            parm_pairs = []  # [[idx:int, val:float], ...]

            for raw_line in lines:
                line = raw_line.strip()
                if not line:
                    continue

                # Tokenize with shlex to keep quoted strings intact
                parts = shlex.split(line)
                if not parts:
                    continue

                key, *values = parts

                # Special handling for PARM rows: "PARM <idx> <value>"
                if key == "PARM" and len(values) >= 2:
                    try:
                        idx = int(values[0])
                        # Some files might include things like '1.0E-03' or '""' afterwards;
                        # we only parse the second token as the numeric coefficient.
                        val = float(values[1])
                        parm_pairs.append([idx, val])
                        # Also expose PARM1, PARM2, ... as individual numeric entries
                        surface_data[f"PARM{idx}"] = val
                    except ValueError:
                        # If parsing fails, just store the raw tokens for inspection
                        surface_data.setdefault("PARM_RAW", []).append(values)
                    # Also keep a generic record of this raw line under the PARM key (optional)
                    surface_data.setdefault("PARMS", []).append(values)
                    continue

                # General case for all other keys: collect as list of lists
                surface_data.setdefault(key, []).append(values)

            # If we collected any numeric PARM pairs, store them under "PARM"
            if parm_pairs:
                # Sort by index to keep a stable order (1..8)
                parm_pairs.sort(key=lambda p: p[0])
                surface_data["PARM"] = parm_pairs

            # Flatten single-occurrence keys (except PARM-related we want to keep as designed)
            for k in list(surface_data.keys()):
                if k in ("PARM", "PARMS") or k.startswith("PARM") and k != "PARM":
                    # Keep "PARM" as list of [idx, val], keep "PARM1/2/..." as floats, keep "PARMS" as a list
                    continue
                v = surface_data[k]
                if isinstance(v, list) and len(v) == 1:
                    surface_data[k] = v[0]

            # Only record a real surface (has both SURF and SSID)
            if "SURF" in surface_data:
                surfaces.append(surface_data)

        self._SurfDict = surfaces
        return self._SurfDict


    def ConvertIntoLens(self):
        """
        Convert the read data into a Lens class object.
        """

        lens = Lens()

        # I cannot guarantee the correctness of these rules, you (whoever not me that's using this) might need to check with the Zemax version you're working with and see if your ZMX files use the same kind of notation.

        for d in self._SurfDict:
            #print("\nNext:\n")
            #self._PrintDict(d)

            if "STOP" in d:
                # Surface type in ZMX tend to be marked with a key "STOP"
                currentS = self._ParseStop(d)
            elif float(d["CURV"][0]) == 0:
                # Start surface is usually object space, which is not of any concern here. Object space will have a clear semi-diameter of infinity, thus a curvature of 0. Use this trait to judge if it's the object space, if so, ski
                continue

            elif d["TYPE"][0] == "STANDARD":
                currentS = self._ParseStandard(d)
            elif d["TYPE"][0] == "EVENASPH":
                currentS = self._ParseEvenasph(d)

            else:
                # Other non stated types can be ignored for now
                continue

            lens.AddSurface(currentS)

        self.lens = lens
        return lens

    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ParseStandard(self, d):

        curvature = float(d["CURV"][0])
        if(curvature == 0):
            radius = INFINITY
        else:
            radius = 1/curvature

        thickness = float(d["DISZ"][0])

        clearSemi = float(d["DIAM"][0])

        if("GLAS" in d):
            material = d["GLAS"][0]
            return Surface(radius, thickness, clearSemi, material)
        else:
            return Surface(radius, thickness, clearSemi)


    def _ParseStop(self, d):

        thickness = float(d["DISZ"][0])

        return Stop(thickness)


    def _ParseEvenasph(self, d):

        curvature = float(d["CURV"][0])
        if (curvature == 0):
            radius = INFINITY
        else:
            radius = 1 / curvature

        thickness = float(d["DISZ"][0])

        clearSemi = float(d["DIAM"][0])

        if "CONI" in d:
            conic = float(d["CONI"][0])
        else:
            # By default, the conic is 0, and CONI is not going to appear
            conic = 0

        coef = []
        for i in d["PARMS"]:
            coef.append(float(i[1]))

        print(conic)
        print(coef)
        if ("GLAS" in d):
            material = d["GLAS"][0]
            return EvenAspheric(radius, thickness, clearSemi, material, conic, coef)
        else:
            # Pass in default material, whatever the Material class define the default is
            return EvenAspheric(radius, thickness, clearSemi, DEFAULT_MAT_NAME, conic, coef)


    def _PrintDictList(self, listOfDict):
        """
        Print a list of dicts.
        """
        for d in listOfDict:
            print("\n\n")
            self._PrintDict(d)


    def _PrintDict(self, d):
        """
        Print a single dict.
        """
        for key, value in d.items():
            print(f"{key}: {value}")


