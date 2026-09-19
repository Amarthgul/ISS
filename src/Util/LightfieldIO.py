"""Portable pseudo light fields containing rays on a shared z reference plane."""

from itertools import chain
from pathlib import Path

import numpy as np

import Util.Backend as Backend
from Raytracing.RayBatch import RayBatch


class LightfieldIO:
    """Read/write RayBatch objects as ``.csv`` or ``.npz`` files.

    Each ray stores nine values: x, y, dx, dy, dz, wavelength (nm), and
    polarization coefficients in RayBatch column order (7, 8, 9): the two
    diagonal terms followed by the ellipse tilt. Coordinates retain their
    input units. The shared reference_z is stored once per file.

    NPZ stores float32 arrays named ``rays`` (N, 9) and ``reference_z`` (scalar).
    CSV stores float32 values as text with nine significant digits, sufficient
    for an exact float32 round trip, rather than as four-byte binary numbers.
    Its first two lines are comments: ``# reference_z,<value>`` and column names.
    Scripts can read the ray table using
    ``np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)``.

    Reading returns an (N, 12) float32 RayBatch on the configured backend.
    Surface index and RGB channel are zero, and there are no AOV columns.
    Original ray origins cannot be recovered; loaded origins lie on the plane.

    Example::

        LightfieldIO.Write(raybatch, "lightfield.csv", referenceZ=10.0)
        recovered = LightfieldIO.Read("lightfield.csv")
    """

    _COLUMNS = "x,y,dx,dy,dz,wavelength,polarization_1,polarization_2,tilt"

    @staticmethod
    def _Path(filePath):
        path = Path(filePath)
        if path.suffix.lower() not in (".csv", ".npz"):
            raise ValueError("Light field file extension must be .csv or .npz.")
        return path

    @staticmethod
    def _Float32(values, name):
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.asarray(values, dtype=np.float32)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain finite float32 values.")
        return result

    @staticmethod
    def _ReferenceZ(value):
        result = LightfieldIO._Float32(value, "reference_z")
        if result.ndim != 0:
            raise ValueError("reference_z must be a scalar.")
        return result

    @staticmethod
    def Write(raybatch, filePath, referenceZ):
        """Project and save rays without modifying the supplied RayBatch.

        Projection uses p + ((referenceZ - p.z) / d.z) * d, including negative
        distances. Directions are preserved, without normalization. Rays
        parallel to the plane are retained at their current x/y if already on
        it; otherwise a ValueError is raised before writing the file. The plane
        is rounded to float32 before projection to match the stored position.
        RayBatch(None) and batches with zero rows are saved as empty fields.

        :return: Path to the written file.
        """
        path = LightfieldIO._Path(filePath)
        reference = LightfieldIO._ReferenceZ(referenceZ)
        if not isinstance(raybatch, RayBatch):
            raise TypeError("raybatch must be a RayBatch object.")

        values = raybatch.value
        if values is None:
            values = np.empty((0, 11))
        elif hasattr(values, "get"):
            values = values.get()  # CuPy arrays must be transferred to the CPU.
        values = np.asarray(values)
        if values.ndim != 2 or values.shape[1] < 11:
            raise ValueError("RayBatch data must have shape (N, 11) or more columns.")

        # Only retained fields need to be numeric and finite; ignore all AOVs.
        values = np.asarray(values[:, :10], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Ray positions, directions and radiance must be finite.")
        offset = float(reference) - values[:, 2]
        parallel = values[:, 5] == 0
        if np.any(parallel & (offset != 0)):
            raise ValueError("Rays parallel to the reference plane cannot be projected.")

        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            distance = np.divide(offset, values[:, 5],
                                 out=np.zeros_like(offset), where=~parallel)
            xy = values[:, :2] + distance[:, None] * values[:, 3:5]
            rays = LightfieldIO._Float32(
                np.concatenate((xy, values[:, 3:10]), axis=1), "Projected rays")

        if path.suffix.lower() == ".npz":
            # A file handle prevents NumPy from appending another extension.
            with path.open("wb") as stream:
                np.savez(stream, rays=rays, reference_z=reference)
        else:
            header = f"reference_z,{float(reference):.9g}\n{LightfieldIO._COLUMNS}"
            np.savetxt(path, rays, delimiter=",", fmt="%.9g", header=header,
                       encoding="utf-8")
        return path

    @staticmethod
    def Read(filePath):
        """Load a light field, restoring z and zeroing surface/RGB indices."""
        path = LightfieldIO._Path(filePath)
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if not {"rays", "reference_z"}.issubset(archive.files):
                    raise ValueError("Light field archive requires rays and reference_z.")
                reference = LightfieldIO._ReferenceZ(archive["reference_z"])
                rays = LightfieldIO._Float32(archive["rays"], "Stored rays")
        else:
            with path.open("r", encoding="utf-8") as stream:
                metadata = stream.readline().strip().split(",")
                if len(metadata) != 2 or metadata[0] != "# reference_z":
                    raise ValueError("Missing CSV reference_z header.")
                reference = LightfieldIO._ReferenceZ(metadata[1])
                if stream.readline().strip() != "# " + LightfieldIO._COLUMNS:
                    raise ValueError("Invalid light field CSV column header.")
                first = stream.readline()
                if first:
                    rays = np.loadtxt(chain((first,), stream), delimiter=",",
                                      dtype=np.float32, ndmin=2)
                    rays = LightfieldIO._Float32(rays, "Stored rays")
                else:
                    rays = np.empty((0, 9), dtype=np.float32)

        if rays.ndim != 2 or rays.shape[1] != 9:
            raise ValueError("Stored rays must have shape (N, 9).")
        values = np.zeros((rays.shape[0], 12), dtype=np.float32)
        values[:, :2] = rays[:, :2]
        values[:, 2] = reference
        values[:, 3:10] = rays[:, 2:9]
        return RayBatch(Backend.backend.asarray(values))
