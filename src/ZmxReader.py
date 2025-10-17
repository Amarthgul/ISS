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

from Surfaces.Surface import Surface
from Surfaces.EvenAspheric import EvenAspheric
from Surfaces.Stop import Stop
from Lens import Lens


class LensFromZmx:
    def __init__(self, path):
        self.path = path
        self.content = None

        self._SurfDict = None # This is a list of dict of all surfaces parsed form the zmx file

        self.ParseFile()


    def ParseFile(self):
        self.ReadFile()
        self.SpiltSurface()
        self.ConvertIntoLens()


    def ReadFile(self):
        with open(self.path, 'r', encoding='utf-16', errors='ignore') as f:
            self.content = f.read()
        return self.content


    def SpiltSurface(self):
        """
        Parses a Zemax .zmx file into a list of per-surface dictionaries.
        Each dict maps field names (e.g. "CURV", "PARM") to lists of their values (as strings).
        """

        # Read and decode the file (UTF-16 is typical for Zemax)

        # Split into surface blocks by "SURF" (keep the marker)
        blocks = re.split(r'(?=^SURF\s)', self.content, flags=re.MULTILINE)
        surfaces = []

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            lines = block.splitlines()
            surface_data = {}

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                key = parts[0]
                values = parts[1:]

                # Append values as a list of strings
                if key in surface_data:
                    surface_data[key].append(values)
                else:
                    surface_data[key] = values

            if("SURF" in surface_data and "SSID" in surface_data):
                # For ZMX file, there will be a lot of headers before the first surface data appears. This checks if the recorded dict has SURF and SSID field, only add into surface dict list if it does.
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
            print("\nNext:\n")
            self._PrintDict(d)

            if "STOP" in d:
                # Surface type in ZMX tend to be marked with a key "STOP"
                self._ParseStop(d)
            elif "GLAS" not in d:
                # Skip other types of surface, this usually applies to OBJECT and IMAGE row in ZMX
                continue

            elif d["TYPE"][0] == "STANDARD":
                self._ParseStandard(d)
            elif d["TYPE"][0] == "EVENASPH":
                self._ParseEvenasph(d)



    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _ParseStandard(self, d):
        pass


    def _ParseStop(self, d):
        pass


    def _ParseEvenasph(self, d):
        pass


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


