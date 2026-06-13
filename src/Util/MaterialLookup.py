

import pandas as pd
import numpy as np
from pathlib import Path
from .Misc import RectPath

MATERIAL_LOOKUP_RESULT_FILENAME = "MaterialLookUprResult.csv"

def ReadSheet(excel_path = RectPath("resources/MaterialRefractionIndices.xlsx")):

    return pd.read_excel(excel_path)


def _WriteMaterialLookupResult(result: pd.DataFrame, writePath):
    output_dir = Path(writePath)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / MATERIAL_LOOKUP_RESULT_FILENAME, index=False)


def FindClosestMaterials(
    df: pd.DataFrame,
    line: str,
    n_target: float,
    v_target: float,
    top_k: int = 1,
    writePath = None
):
    """
    Given a refractive index n_target and Abbe number v_target for one
    of the Fraunhofer lines {d, D, e}, find the closest match(es) in
    the provided Excel table.

    :param df: Dataframe object of the sheet containing all the n and V.
    :param line: One of {'d', 'D', 'e'}, indicating which Fraunhofer line
                 the n and V are for.
    :param n_target: Target refractive index at the specified line.
    :param v_target: Target Abbe number at the specified line.
    :param top_k: Number of closest matches to return (default=1).
    :param writePath: Optional directory path where the result CSV will be written.
    :return: A DataFrame of the closest match(es).
    """


    line_map = {
        'd':  ('d', 'V_d'),
        'D':  ('D', 'V_D'),
        'e':  ('e', 'V_e')
    }


    n_Range = [1.4, 2]
    n_Delta = n_Range[1] - n_Range[0]
    V_Range = [20, 90]
    V_Delta = V_Range[1] - V_Range[0]

    index_col, abbe_col = line_map[line]

    # 3. Precompute the normalized target values for n and V
    n_target_norm = (n_target - n_Range[0]) / n_Delta
    v_target_norm = (v_target - V_Range[0]) / V_Delta

    # 4. Compute distance in the normalized space for each row
    def distance(row):
        n_val = row[index_col]
        v_val = row[abbe_col]

        # If any row is missing data, handle or skip
        if pd.isna(n_val) or pd.isna(v_val):
            return np.inf

        n_norm = (n_val - n_Range[0]) / n_Delta
        v_norm = (v_val - V_Range[0]) / V_Delta

        return np.sqrt((n_norm - n_target_norm) ** 2 + (v_norm - v_target_norm) ** 2)

    df["distance"] = df.apply(distance, axis=1)

    # 5. Sort by ascending distance and grab the top_k
    df_sorted = df.sort_values("distance", ascending=True).head(top_k)

    # 6. Create a new DataFrame with only the desired columns
    result = pd.DataFrame({
        "Cate": df_sorted["Cate"],
        "Name": df_sorted["Name"],
        f"actual n_{line}": df_sorted[index_col],
        f"actual V_{line}": df_sorted[abbe_col],
        f"desired n_{line}": [n_target] * len(df_sorted),
        f"desired V_{line}": [v_target] * len(df_sorted),
    }).reset_index(drop=True)

    if writePath is not None:
        _WriteMaterialLookupResult(result, writePath)

    return result


def FindClosestMaterialsBatch(
    df: pd.DataFrame,
    line: str,
    targets,
    top_k: int = 1,
    writePath = None
):
    """
    Find the closest material matches for a list of refractive index and
    Abbe number targets.

    :param df: Dataframe object of the sheet containing all the n and V.
    :param line: One of {'d', 'D', 'e'}, indicating which Fraunhofer line
                 the n and V are for.
    :param targets: List of target pairs in the form [[n, V], ...].
    :param top_k: Number of closest matches to return for each target.
    :param writePath: Optional directory path where the combined result CSV will be written.
    :return: A DataFrame of the closest matches for all target pairs.
    """

    results = []

    for target_index, (n_target, v_target) in enumerate(targets):
        result = FindClosestMaterials(df, line, n_target, v_target, top_k=top_k)
        result.insert(0, "Target", target_index)
        results.append(result)

    if results:
        batch_result = pd.concat(results, ignore_index=True)
    else:
        batch_result = pd.DataFrame()

    if writePath is not None:
        _WriteMaterialLookupResult(batch_result, writePath)

    return batch_result


if __name__ == "__main__":
    pass
