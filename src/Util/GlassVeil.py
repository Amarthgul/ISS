

import math

import matplotlib.pyplot as plt

import Material as material_module
from Material import Material
from Util.Globals import LambdaLines



plt.rcParams['savefig.dpi'] = 300


def PlotGlassVeil(
    catalogue=["SCHOTT"],
    materials=["BK7"],
    line="d",
    nRange=[1.4, 2],
    VRange=[20, 90],
    pointSizeGrowthRatio=1.0,
    pointSizeMax=128,
):
    """Plot an Abbe diagram for the requested glass catalogues.

    ``line`` selects the Fraunhofer reference line and supports ``"d"``,
    ``"D"``, and ``"e"``.  When ``materials`` is non-empty, points and labels
    whose names are not in that list are faded.  The returned figure and axis
    can be further styled or saved by the caller.  ``pointSizeGrowthRatio``
    controls how quickly markers grow as the ranges narrow, while
    ``pointSizeMax`` caps their area in points squared.
    """
    noneHighlightOpacity = 0.15

    nRangeDefault = [1.4, 2]
    VRangeDefault = [20, 90]
    pointSizeDefault = 24
    textSizeDefault = 6

    nSpan = abs(nRange[1] - nRange[0])
    VSpan = abs(VRange[1] - VRange[0])
    if nSpan == 0 or VSpan == 0:
        raise ValueError("nRange and VRange must each span more than zero.")
    if pointSizeGrowthRatio < 0:
        raise ValueError("pointSizeGrowthRatio cannot be negative.")
    if pointSizeMax <= 0:
        raise ValueError("pointSizeMax must be greater than zero.")

    nZoom = max(1.0, abs(nRangeDefault[1] - nRangeDefault[0]) / nSpan)
    VZoom = max(1.0, abs(VRangeDefault[1] - VRangeDefault[0]) / VSpan)
    rangeZoom = math.sqrt(nZoom * VZoom)
    pointSize = min(
        pointSizeDefault * rangeZoom ** pointSizeGrowthRatio,
        pointSizeMax,
    )
    # Matplotlib scatter sizes are areas, while font sizes are linear.  The
    # square root therefore keeps labels growing at the marker's visual rate.
    textSize = textSizeDefault * math.sqrt(pointSize / pointSizeDefault)


    if isinstance(catalogue, str):
        catalogue = [catalogue]
    if isinstance(materials, str):
        materials = [materials]

    lineMethods = {
        "d": (Material.n_d, Material.V_d),
        "D": (lambda material: material.RI(LambdaLines["D"]), Material.V_D),
        "e": (Material.n_e, Material.V_e),
    }
    if line not in lineMethods:
        raise ValueError('line must be one of "d", "D", or "e".')

    requestedCatalogues = {str(name).upper() for name in catalogue}
    highlightedMaterials = {str(name).upper() for name in materials}
    glassTable = material_module.GlassTable
    if glassTable is None:
        glassTable = material_module.pd.read_excel(material_module.GlassTablePath)

    catalogueNames = glassTable["Cate"].astype(str).str.upper()
    selectedRows = glassTable[catalogueNames.isin(requestedCatalogues)]
    selectedRows = selectedRows.drop_duplicates(subset=["Cate", "Name"])
    if selectedRows.empty:
        raise ValueError("None of the requested catalogues were found in the glass table.")

    indexMethod, abbeMethod = lineMethods[line]
    plotRows = []
    originalGlassTable = material_module.GlassTable
    try:
        for _, row in selectedRows.iterrows():
            # Material looks up by name only.  Restricting its table to this row
            # ensures duplicate glass names use the requested catalogue data.
            material_module.GlassTable = glassTable[
                (glassTable["Cate"] == row["Cate"])
                & (glassTable["Name"] == row["Name"])
            ]
            material = Material(row["Name"])
            refractiveIndex = float(indexMethod(material))
            abbeNumber = float(abbeMethod(material))
            if math.isfinite(refractiveIndex) and math.isfinite(abbeNumber):
                plotRows.append((row["Cate"], row["Name"], abbeNumber, refractiveIndex))
    finally:
        material_module.GlassTable = originalGlassTable

    if not plotRows:
        raise ValueError("The selected catalogues contain no finite Abbe data.")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    cataloguesInPlot = list(dict.fromkeys(row[0] for row in plotRows))
    colourMap = plt.get_cmap("tab10")

    for catalogueIndex, catalogueName in enumerate(cataloguesInPlot):
        colour = colourMap(catalogueIndex % colourMap.N)
        catalogueRows = [row for row in plotRows if row[0] == catalogueName]
        for pointIndex, (_cate, name, abbeNumber, refractiveIndex) in enumerate(catalogueRows):
            highlighted = not highlightedMaterials or str(name).upper() in highlightedMaterials
            opacity = 1.0 if highlighted else noneHighlightOpacity
            ax.scatter(
                abbeNumber,
                refractiveIndex,
                s=pointSize,
                color=colour,
                alpha=opacity,
                label=catalogueName if pointIndex == 0 else None,
                zorder=3 if highlighted else 2,
            )
            ax.annotate(
                str(name),
                (abbeNumber, refractiveIndex),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=textSize,
                color=colour,
                alpha=opacity,
                zorder=3 if highlighted else 2,
            )

    # Abbe diagrams conventionally run from high V (left) to low V (right).
    ax.set_xlim(VRange[1], VRange[0])
    ax.set_ylim(nRange)
    ax.set_xlabel(f"Abbe number V{line}")
    ax.set_ylabel(f"Index of refraction n{line}")
    ax.set_title("Abbe Diagram")
    ax.grid(True, which="major", linewidth=0.7, alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.12)
    ax.legend(title="Catalogue")

    if plt.get_backend().lower() != "agg":
        plt.show()

    return fig, ax






def main():
    PlotGlassVeil()

if __name__ == "__main__":
    main()
