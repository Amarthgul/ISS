
# For quickly converting Zemax optimization conditions into operands

import pandas as pd
from pathlib import Path
from Util.Misc import RectPath

CURVATURE_OPERANDS_FILENAME = "CurvatureOperands.csv"

def _WriteCurvatureOperands(result: pd.DataFrame, writePath=RectPath("resources/")):
    output_dir = Path(writePath)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / CURVATURE_OPERANDS_FILENAME, index=False)


def RadiToCurv(RadiPairs, writeFile=True):
    """
    Convert some pairs of radii conditions into Zemax curvature operands.

    :param RadiPairs: list of pairs of radii in the form of [[r1t, r1b], [r2t, r2b], ...], each entry in a pair represents top and bottom bounds.
    :param writeFile: whether to write the resulting Zemax curvature into a cvs file.

    :return: Zemax curvature operands of CVGT and CVLT.
    """

    operands = []

    for surface_index, radi_pair in enumerate(RadiPairs, start=1):
        if len(radi_pair) != 2:
            raise ValueError("Each radius condition must contain exactly two bounds.")

        curvatures = []
        for radius in radi_pair:
            if radius == 0:
                raise ValueError("Cannot convert a radius of 0 to curvature.")

            curvatures.append(1 / radius)

        lower_curvature = min(curvatures)
        upper_curvature = max(curvatures)

        operands.append({
            "Surface": surface_index,
            "Operand": "CVGT",
            "Target": lower_curvature,
        })
        operands.append({
            "Surface": surface_index,
            "Operand": "CVLT",
            "Target": upper_curvature,
        })

    result = pd.DataFrame(operands, columns=["Surface", "Operand", "Target"])

    if writeFile:
        _WriteCurvatureOperands(result)

    return result





