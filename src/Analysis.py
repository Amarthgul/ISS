

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
            w=self.imager.width,
            h=self.imager.height,
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

            currentCent = Centroid(image)

            # Real centroid position (imager coordinates; optical axis is (0,0))
            cRatioX, cRatioY = currentCent.centroidPositionRatio
            cx, cy = cRatioX*self.imager.width, cRatioY*self.imager.height


            # Compute radial distortion (fraction)
            r_max = math.hypot(self.imager.width, self.imager.height)
            r_id = math.hypot(x_id, y_id)
            r_rl = math.hypot(float(cx), float(cy))
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
                "mapping": "rectilinear (r = f*tan(theta))",
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

