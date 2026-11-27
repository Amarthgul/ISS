"""Compare image-plane spot positions for corresponding field, pupil and spectral samples."""

from copy import copy, deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Raytracing.Emission import EmitField
from Raytracing.RayBatch import GenerateEmpty, RayBatch
from Util.Backend import backend as bd
from Util.Globals import LambdaLines
from Util.ImageIO import SaveAsEXR
from Util.Misc import NumpyConversion, RectPath
from Util.Sampling import PolarDistribution


class DualSystem:
    """Sparse spot comparison at fieldAngles = [horizontal, vertical] in degrees.

    objectDistance and focusDistance are in mm, with one entry per system.
    A None fNumber uses that lens's configured aperture.

    pupilSampleCount is a target rounded up to complete polar rings. wavelengths
    is a sequence in nm. The closest of C, e and F selects the diagnostic RGB
    channel. Each field/pupil/wavelength combination has one ID on both sides.
    """


    def __init__(self, systemOne=None, systemTwo=None):
        self.systemOne = systemOne
        self.systemTwo = systemTwo

        self.objectDistance = [1350, 1350]
        self.focusDistance = [13500, 13500]
        self.fNumber = [None, None]

        self.fieldAngles = [0, 0]
        self.pupilSampleCount = 25
        self.matchCentroids = True  # Align the second spot along the field direction in the plot.

        self.wavelengths = tuple(LambdaLines[line] for line in ("e", "F", "C"))

        self._arrowSize = 1  # Arrowhead scale relative to the standard quiver head.
        self._arrowLine = 0.002  # Shaft width as a fraction of axes width; None uses autosizing.
        self._plotDPI = 300


    def Render(self, fileName=None):
        """Trace each configured system once at its own object distance, focus and aperture.

        Return paths to the two RGB EXRs and the shift PNG. fileName is a stem,
        relative to resources/Results or absolute.
        """
        return self._Render(False, fileName or "RayShift")


    def RenderParaxial(self, fileName=None):
        """Compare system one's Gaussian and geometric images using entry zero for both."""
        return self._Render(True, fileName or "RayShiftParaxial")


    def _Launch(self, lens, unitPupil, indexed, objectDistance):
        """Map a copy of the indexed sample grid onto this lens's working pupil."""
        targets, keep = lens.entrancePupil.MapNormalizedSamples(unitPupil)
        rows = indexed.value.reshape(len(targets), len(self.wavelengths), -1)[keep].copy()
        launch = EmitField(*self.fieldAngles, objectDistance, targets[keep])
        rows[:, :, :6] = launch.value[:, None, :6]
        return RayBatch(rows.reshape(-1, indexed.value.shape[1]))


    def _Render(self, idealFirst, fileName):
        pupil = PolarDistribution(1., self.pupilSampleCount, radialExponent=0.75)
        count = len(pupil) * len(self.wavelengths)
        lines = bd.asarray(self.wavelengths)
        rgbLines = bd.asarray([LambdaLines[k] for k in ("C", "e", "F")])
        channels = bd.abs(lines[:, None] - rgbLines).argmin(axis=1)
        indexed = GenerateEmpty(count)
        indexed.value[:, 6] = bd.tile(lines, len(pupil))
        indexed.SetChannel(bd.tile(channels, len(pupil)))
        indexed.AppendAOV(bd.arange(count))

        def prepare(system, index, gaussianFocus):
            lens = system.lens
            if self.fNumber[index] is not None:
                lens = deepcopy(lens)
                lens.SetAperture(self.fNumber[index])
            focus = self.focusDistance[index]
            bfd = lens.ParaxialBFD(focus) if gaussianFocus else lens.BestFocusBFD(focus)
            detector = copy(system.imager)
            detector.SetBFD(bfd)
            detector.SetLensLength(lens.totalAxialLength)
            launch = self._Launch(lens, pupil, indexed, self.objectDistance[index])
            return lens, detector, launch

        lensA, detectorA, launchA = prepare(self.systemOne, 0, idealFirst)
        if idealFirst:
            lensB, detectorB, launchB = lensA, copy(detectorA), launchA.Copy()
            matrices = lensA.ParaxialSurfaceMatrices()
        else:
            lensB, detectorB, launchB = prepare(self.systemTwo, 1, False)
            matrices = None

        imageA, idsA, xyA = self._Capture(lensA, detectorA, launchA, matrices)
        imageB, idsB, xyB = self._Capture(lensB, detectorB, launchB, None)
        ids, a, b = np.intersect1d(idsA, idsB, assume_unique=True, return_indices=True)
        counts = len(ids), len(idsA) - len(ids), len(idsB) - len(ids)

        names = ("Paraxial", "Geometric") if idealFirst else ("SystemOne", "SystemTwo")
        stem = Path(RectPath("resources/Results")) / fileName
        paths = []
        for name, image in zip(names, (imageA, imageB)):
            base = stem.name + "_" + name
            SaveAsEXR(image, str(stem.parent), base, flipVert=True, rotate=True)
            paths.append(stem.parent / (base + ".exr"))

        plot = stem.parent / (stem.name + "_Shift.png")
        start, end = xyA[a], xyB[b]
        offset = None
        if len(ids):
            direction = np.tan(np.deg2rad(self.fieldAngles))
            norm = np.linalg.norm(direction)
            if norm:
                direction /= norm
            offset = (end - start).mean(axis=0) @ direction
            if self.matchCentroids:
                end -= offset * direction
        self._Plot(plot, names, start, end, ids, NumpyConversion(channels), counts, offset)
        return paths[0], paths[1], plot


    def _Capture(self, lens, detector, rays, matrices):
        if not rays.IsNoneType():
            if matrices is None:
                rays = lens.Propagate(rays, reflection=False)[0]
                if not rays.IsNoneType():
                    rays = detector.DetectorHits(rays)
            else:
                rays = lens.PropagateParaxial(rays, matrices)
                if not rays.IsNoneType():
                    rays = detector.IntersectParaxial(rays)[0]

        if rays.IsNoneType():
            return detector.AcquireEmpty(), np.empty(0, dtype=int), np.empty((0, 2))

        # Match the imager's half-open pixel footprint, including odd-sized detectors.
        pitch = detector.width / detector.horizontalPx
        size = bd.asarray([detector.horizontalPx, detector.verticalPx])
        pixels = rays.value[:, :2] / pitch + size / 2
        rays.Mask(bd.all((pixels >= 0) & (pixels < size), axis=1))
        ids = NumpyConversion(rays.GetAoV()[:, 0]).astype(int)
        xy = NumpyConversion(rays.Position()[:, :2])
        # The correspondence ID has no radiometric meaning. Deposit RGB only.
        rays.value = rays.value[:, :12]
        image = detector.IntegralHits(rays, overExpNoiseRemoval=None)
        return image, ids, xy


    def _Plot(self, path, names, start, end, ids, channels, counts, offset):
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        colors = ("tab:red", "tab:green", "tab:blue")
        for i, wavelength in enumerate(self.wavelengths):
            mask = ids % len(self.wavelengths) == i
            origin, target = start[mask], end[mask]
            color = colors[channels[i]]
            ax.scatter(*origin.T, s=10, color=color, label=f"{wavelength:g} nm")
            if len(origin):
                ax.quiver(*origin.T, *(target - origin).T, color=color, angles="xy",
                          scale_units="xy", scale=1, minlength=0, width=self._arrowLine,
                          headwidth=3 * self._arrowSize, headlength=5 * self._arrowSize,
                          headaxislength=4.5 * self._arrowSize)
                ax.update_datalim(target)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        offsetText = "unavailable (no paired hits)" if offset is None else f"{offset:+.6g} mm"
        alignment = "on" if self.matchCentroids else "off"
        ax.set(xlabel="Sensor X (mm)", ylabel="Sensor Y (mm)",
               title=f"{names[0]} → {names[1]}\n{counts[0]} paired hits · "
                     f"{counts[1]} {names[0]} only · {counts[2]} {names[1]} only\n"
                     f"Field-direction centroid offset {offsetText} · alignment {alignment}")
        ax.legend()
        fig.savefig(path, dpi=self._plotDPI)
        plt.close(fig)
