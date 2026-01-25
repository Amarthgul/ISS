

"""
This class is the abridged equivalence of the Zemax Analysis tab.
"""

from Lens import Lens
from Util.Centroid import Centroid
from ObjectSpace.Points import PointsSource
from Util.Backend import backend as bd
import math

class Analysis():
    def __init__(self, lens, imager):
        self.lens = lens
        self.imager = imager

        self.imagerNominalHeight = 24
        self.imagerNominalWidth = 36

        self.distortionData = None


    def Distortion(self, objectT, maxFieldAngleX=0, maxFieldAngleY=0, samplePoints=10):
        """
        Calculate the geometric distortion of the system.

        :param objectT: object distance.
        :param focusT: focus distance.
        :param maxFieldAngleX: maximum field angle along X direction.
        :param maxFieldAngleY: maximum field angle along Y direction.
        :param samplePoints: number of spots used to sample.
        """


        # If user doesn't provide max field angles, fall back to lens AoV (half angles).
        # NOTE: GetAoV returns degrees when unitInDegrees=True.
        aovHori, aovVert, aovDiag = self.lens.GetAoV(
            unitInDegrees=True,
            halfAngle=True,
            w=self.imagerNominalWidth,
            h=self.imagerNominalHeight,
        )

        if maxFieldAngleX is None or maxFieldAngleX == 0:
            maxFieldAngleX = aovHori
        if maxFieldAngleY is None or maxFieldAngleY == 0:
            maxFieldAngleY = aovVert

        # Already traced effective focal length (paraxial EFL)
        EFL = self.lens.focalLength

        # Axial position of the *front* principal plane (z in your coordinate frame)
        # Your convention: object space is -Z, image is +Z, first surface vertex at z=0.
        H = self.lens.frontPrincipalPlane.GetInnerZ()

        # Samples per field point (entrance pupil samples)
        targetSample = 10240


        thetaX_deg = bd.linspace(0.0, float(maxFieldAngleX), int(samplePoints))
        thetaY_deg = bd.linspace(0.0, float(maxFieldAngleY), int(samplePoints))

        # ----------------------------
        # Core computation
        # ----------------------------
        # Output arrays
        fieldX_deg = []
        fieldY_deg = []
        idealX = []
        idealY = []
        realX = []
        realY = []
        distortion = []  # fractional (not %) by default

        # Precompute pupil sample points once
        pupil_pts = self.lens.entrancePupil.GetSamplePoints(targetSample)

        # Distance from object point (z=-objectT) to front principal plane (z=H)
        # Used to place object points at a given field angle for finite object distances.
        obj_to_H = float(objectT) + float(H)


        for tx, ty in zip(thetaX_deg, thetaY_deg):
            tx_f = float(tx)
            ty_f = float(ty)
            tx_rad = math.radians(tx_f)
            ty_rad = math.radians(ty_f)

            # --- Ideal mapping (rectilinear): x_ideal = f * tan(theta_x), y_ideal = f * tan(theta_y)
            x_id = float(EFL) * math.tan(tx_rad)
            y_id = float(EFL) * math.tan(ty_rad)

            # --- Build an object point at distance objectT (object side is -Z)
            # Place the point so that, measured from the front principal plane, it subtends (tx,ty).
            x_obj = obj_to_H * math.tan(tx_rad)
            y_obj = obj_to_H * math.tan(ty_rad)
            z_obj = -float(objectT)

            # Construct PointsSource
            ps = PointsSource(bd.array([[x_obj, y_obj, z_obj, 1, 1, 1]]))


            # Emit rays toward entrance pupil samples
            mainRB = ps.EmitSamplesToward(pupil_pts, 1, addSecondary=9)

            mainRB, mainRP, _ = self.lens.Propagate(mainRB)

            # Form spot on imager
            mainRB, _tir, _vig = self.imager.IntersectRays(mainRB)
            image = self.imager.IntegralRays(mainRB)

            # from Util.ImageIO import ImageConversion, CleanDisplay
            # import matplotlib.pyplot as plt
            # CleanDisplay(image.get() * 100.0)
            # plt.draw()
            # plt.pause(5)

            currentCent = Centroid(image)

            # Real centroid position in mm on the *actual* imager (so enlarging it works)
            cRatioX, cRatioY = currentCent.centroidPositionRatio
            cx = float(cRatioX) * float(self.imager.width)
            cy = float(cRatioY) * float(self.imager.height)

            # --- Normalize distortion by NOMINAL half-diagonal (since cx,cy are centered coords)
            r_max = math.hypot(0.5 * float(self.imagerNominalWidth),
                               0.5 * float(self.imagerNominalHeight))

            r_id = math.hypot(x_id, y_id)
            r_rl = math.hypot(cx, cy)

            if r_id == 0.0:
                d = 0.0
            else:
                d = (r_rl - r_id) / r_max

            fieldX_deg.append(tx_f)
            fieldY_deg.append(ty_f)
            idealX.append(x_id)
            idealY.append(y_id)
            realX.append(float(cx))
            realY.append(float(cy))
            distortion.append(d)

        self.distortionData = {
            "fieldAngleX_deg": fieldX_deg,
            "fieldAngleY_deg": fieldY_deg,
            "idealXY": list(zip(idealX, idealY)),
            "realXY": list(zip(realX, realY)),
            "distortion": distortion,
            "distortion_percent": [x * 100.0 for x in distortion],
            "meta": {
                "EFL": float(EFL),
                "frontPrincipalPlaneZ": float(H),
                "maxFieldAngleX_deg": float(maxFieldAngleX),
                "maxFieldAngleY_deg": float(maxFieldAngleY),
                "samplePoints": int(samplePoints),
                "targetSamplePerField": int(targetSample),
                "nominalWidth": float(self.imagerNominalWidth),
                "nominalHeight": float(self.imagerNominalHeight),
                "normalization": "r_max = nominal half-diagonal",
                "aovBasis": "nominal imager size"
            },
        }

        return [x * 100.0 for x in distortion]


    def PlotDistortionPercentage(self):
        import matplotlib.pyplot as plt
        import numpy as np

        dd = self.distortionData

        # X axis: distortion in percent
        x = np.asarray(dd.get("distortion_percent", []), dtype=float)

        # Y axis: field angle in degrees
        # Use a radial field angle so this works for diagonal sweeps or 2D grids.
        fx = np.asarray(dd.get("fieldAngleX_deg", []), dtype=float)
        fy = np.asarray(dd.get("fieldAngleY_deg", []), dtype=float)
        if fx.size == 0 and fy.size == 0:
            raise RuntimeError("distortionData missing fieldAngleX_deg / fieldAngleY_deg.")

        if fx.size == 0:
            y = np.abs(fy)
        elif fy.size == 0:
            y = np.abs(fx)
        else:
            y = np.sqrt(fx * fx + fy * fy)

        # Sort by field angle (so the curve draws monotonically in Y)
        order = np.argsort(y)
        y = y[order]
        x = x[order]

        # Choose symmetric x-limits around 0 (like optical design software)
        max_abs = float(np.max(np.abs(x))) if x.size else 1.0
        if max_abs < 1e-9:
            max_abs = 1.0
        pad = 0.10 * max_abs
        xlim = (-max_abs - pad, max_abs + pad)

        # Plot
        plt.figure(figsize=(6.0, 6.5), dpi=150)
        plt.plot(x, y, linewidth=2.0)

        # Center 0% line
        plt.axvline(0.0, linewidth=1.2)

        # Grid styling (major + minor gives the “analysis plot” feel)
        plt.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.35)
        plt.minorticks_on()
        plt.grid(True, which="minor", linestyle="-", linewidth=0.5, alpha=0.15)

        plt.xlim(*xlim)
        # Field from 0 to max
        plt.ylim(0.0, float(np.max(y)) if y.size else 1.0)

        plt.xlabel("Percent")
        plt.ylabel("Field Angle (deg)")
        plt.title("Distortion")

        plt.tight_layout()
        plt.show()


    def PlotThroughFocusDistortion(self, objectTs, maxFieldAngleX=0, maxFieldAngleY=0, samplePoints=10):
        """
        Plot multiple distortion curves (distortion % vs field angle) for different object distances.

        :param objectTs: iterable of object distances (same units your system uses; object is at z=-objectT)
        :param maxFieldAngleX: max field angle X (deg), 0 -> auto from lens AoV
        :param maxFieldAngleY: max field angle Y (deg), 0 -> auto from lens AoV
        :param samplePoints: number of field samples per curve
        """
        import matplotlib.pyplot as plt
        import numpy as np

        distortionDataSet = []

        # --- Compute distortion for each object distance
        for objectDistance in objectTs:
            # IMPORTANT: correct argument order
            self.Distortion(objectDistance, maxFieldAngleX=maxFieldAngleX, maxFieldAngleY=maxFieldAngleY,
                            samplePoints=samplePoints)

            dd = self.distortionData

            # X axis: distortion in percent
            x = np.asarray(dd.get("distortion_percent", []), dtype=float)
            if x.size == 0:
                x = np.asarray(dd.get("distortion", []), dtype=float) * 100.0

            # Y axis: radial field angle in degrees (robust for diagonal or 2D sampling)
            fx = np.asarray(dd.get("fieldAngleX_deg", []), dtype=float)
            fy = np.asarray(dd.get("fieldAngleY_deg", []), dtype=float)

            if fx.size == 0:
                y = np.abs(fy)
            elif fy.size == 0:
                y = np.abs(fx)
            else:
                y = np.sqrt(fx * fx + fy * fy)

            # Sort by field angle so each curve is drawn bottom->top cleanly
            order = np.argsort(y)
            y = y[order]
            x = x[order]

            distortionDataSet.append((float(objectDistance), x, y))

        # --- Determine global plot bounds (consistent axes across curves)
        all_x = np.concatenate([d[1] for d in distortionDataSet])
        max_abs = float(np.max(np.abs(all_x))) if all_x.size else 1.0
        if max_abs < 1e-9:
            max_abs = 1.0
        pad = 0.10 * max_abs
        xlim = (-max_abs - pad, max_abs + pad)

        all_y = np.concatenate([d[2] for d in distortionDataSet])
        ymax = float(np.max(all_y)) if all_y.size else 1.0

        # --- Plot
        plt.figure(figsize=(6.0, 6.5), dpi=150)

        for objectDistance, x, y in distortionDataSet:
            # Label with object distance; you can customize formatting if you have units
            plt.plot(x, y, linewidth=1.0, label=f"ObjectT={objectDistance:g}")

        # Center 0% line
        plt.axvline(0.0, linewidth=0.5, c="k")

        # Grid styling
        plt.grid(True, which="major", linestyle="-", linewidth=0.8, alpha=0.35)
        plt.minorticks_on()
        plt.grid(True, which="minor", linestyle="-", linewidth=0.5, alpha=0.15)

        plt.xlim(*xlim)
        plt.ylim(0.0, ymax)

        plt.xlabel("Percent")
        plt.ylabel("Field Angle (deg)")
        plt.title("Through-Focus Distortion")

        # Legend (many curves -> keep it compact)
        plt.legend(loc="best", fontsize=8, framealpha=0.85)

        plt.tight_layout()
        plt.show()


