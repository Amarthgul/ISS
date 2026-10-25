

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from .Misc import RectPath, NumpyConversion
from . import Globals

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


def FindClosestMaterialsByLines(
    df: pd.DataFrame,
    lines,
    targets,
    top_k: int = 1,
    writePath = None
):
    """Find top_k materials for each set of indices at two or more lines.

    ``lines`` contains keys in Globals.LambdaLines, e.g. ['d', "G'"].
    Only Cate and Name are required in df. Indices are calculated using
    Material.RI at the corresponding wavelengths in nanometers, using
    the dispersion catalog rather than precomputed spreadsheet indices.
    Each row of ``targets`` supplies indices in that same order. Each
    index uses the existing n range [1.4, 2.0] for normalization; distance
    is sqrt(sum(((actual - desired) / (2.0 - 1.4)) ** 2)). Values outside
    this range are allowed and are not clipped.

    Rows missing a finite index at any requested line are excluded.
    Materials absent from the dispersion catalog are skipped with a warning.
    Results contain a zero-based Target, Cate, Name, distance, and actual
    and desired indices for every line. The input DataFrame is unchanged.
    ``writePath`` optionally saves the combined result CSV.
    """
    lines = list(lines)
    if len(lines) < 2 or len(set(lines)) != len(lines):
        raise ValueError("lines must contain at least two distinct line names")
    missing = [line for line in lines if line not in Globals.LambdaLines]
    if missing:
        raise ValueError(f"Unknown lines in Globals.LambdaLines: {missing}")
    wavelengths = np.asarray([Globals.LambdaLines[line] for line in lines], dtype=float)
    if not np.isfinite(wavelengths).all() or (wavelengths <= 0).any():
        raise ValueError("Line wavelengths must be finite positive nanometer values")
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer")

    targets = list(targets)
    target_values = np.asarray(targets, dtype=float)
    if targets and (target_values.ndim != 2 or target_values.shape[1] != len(lines)):
        raise ValueError("Each target must contain one index per requested line")
    if not np.isfinite(target_values).all():
        raise ValueError("Target indices must be finite numbers")

    columns = ["Target", "Cate", "Name", "distance"]
    for line in lines:
        columns.extend([f"actual n_{line}", f"desired n_{line}"])

    # Import lazily so the ordinary spreadsheet lookup does not load the
    # dispersion catalog. Evaluate all lines once per material for the batch.
    from Material import Material, GlassTable

    values = np.empty((len(df), len(lines)), dtype=float)
    values.fill(np.nan)
    if targets:
        catalog_pairs = set(GlassTable[["Cate", "Name"]].itertuples(index=False, name=None))
        unavailable = []
        for position, (category, name) in enumerate(df[["Cate", "Name"]].itertuples(index=False, name=None)):
            if (category, name) not in catalog_pairs:
                unavailable.append((category, name))
                continue
            material = Material(name, category=category)
            values[position] = NumpyConversion(material.RI(wavelengths))
        if unavailable:
            warnings.warn(
                f"Skipped {len(unavailable)} materials absent from the dispersion catalog: {unavailable}",
                RuntimeWarning, stacklevel=2)
    valid = np.isfinite(values).all(axis=1)
    candidates = df.loc[valid]
    values = values[valid]
    results = []
    for target_index, target in enumerate(target_values):
        distances = np.sqrt(np.sum(((values - target) / (2.0 - 1.4)) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")[:top_k]
        matches = candidates.iloc[order]
        result = matches[["Cate", "Name"]].reset_index(drop=True).copy()
        result.insert(0, "Target", target_index)
        result["distance"] = distances[order]
        for line, desired in zip(lines, target):
            result[f"actual n_{line}"] = values[order, lines.index(line)]
            result[f"desired n_{line}"] = desired
        results.append(result)

    result = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=columns)
    if writePath is not None:
        _WriteMaterialLookupResult(result, writePath)
    return result


if __name__ == "__main__":
    pass
