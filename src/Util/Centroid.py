from .Backend import backend as bd

class Centroid:
    def __init__(self, spot):
        """RGB array of the spot"""
        self.value = bd.array(spot)

        """Defines the maximum field angle this spot could reach, i.e. the field angle at the very corner"""
        self.diagonalFieldAngle = 22

        """Defines the optical axis location, i.e., the 0 degree field on the spot"""
        self.centerRatio = [.5, .5]

        """Weight of the RGB channels when doing the sum"""
        self.channelWeight = [1, 1, 1]

        """
        centroidAngle is stored as:
            (angle_x_deg, angle_y_deg, angle_radial_deg)

        where angle_x/y are the field angle components relative to optical axis,
        and angle_radial is sqrt(angle_x^2 + angle_y^2).
        """
        self.centroidAngle = None

        """X and Y position of the centroid as a ratio of the entire imager, assuming middle of the image is (0, 0)"""
        self.centroidPositionRatio = None


        self.Update()




    def Update(self):

        I = self._weighted_intensity()

        H, W = int(I.shape[0]), int(I.shape[1])

        # Optical axis pixel location from centerRatio
        # Use (W-1, H-1) so that ratios align with pixel grid endpoints
        x0 = self.centerRatio[0] * (W - 1)
        y0 = self.centerRatio[1] * (H - 1)

        # Total weight (energy)
        sumI = bd.sum(I)

        # If empty / zero energy, define centroid at optical axis
        # (Also handles NaN/Inf by falling back if sumI isn't finite)
        sumI_f = self._to_float(sumI)
        if not (sumI_f > 0.0):
            xc = x0
            yc = y0
        else:
            # Compute centroid in pixel coordinates
            xs = bd.arange(W).reshape(1, W)
            ys = bd.arange(H).reshape(H, 1)

            xc = bd.sum(I * xs) / sumI
            yc = bd.sum(I * ys) / sumI

            # Convert to Python floats for downstream usage / printing
            xc = self._to_float(xc)
            yc = self._to_float(yc)


        # Pixel displacement from optical axis
        dx = xc - x0
        dy = yc - y0

        self.centroidPositionRatio = (dx / W, dy / H)


        # Compute the maximum possible radius (distance to the farthest corner) from the optical axis
        rx = max(x0, (W - 1) - x0)
        ry = max(y0, (H - 1) - y0)
        rmax = (rx * rx + ry * ry) ** 0.5

        if rmax <= 0:
            ax = 0.0
            ay = 0.0
        else:
            # Linear mapping: farthest corner => diagonalFieldAngle
            scale = self.diagonalFieldAngle / rmax
            ax = dx * scale
            ay = dy * scale

        ar = (ax * ax + ay * ay) ** 0.5

        # assign centroidAngle
        self.centroidAngle = (float(ax), float(ay), float(ar))


    def CentroidAngle(self):
        return self.centroidAngle


    def Override(self, newSpot, fieldAngle=22):
        self.value = bd.array(newSpot)
        self.diagonalFieldAngle = fieldAngle
        self.Update()


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _to_float(self, x):
        """Convert bd scalar (numpy or cupy) to Python float."""
        try:
            return float(x.item())
        except Exception:
            return float(x)


    def _weighted_intensity(self):
        """
        Return a 2D scalar intensity map from the spot.
        - If input is HxWx3: apply channelWeight and sum channels.
        - If input is already HxW: return as-is.
        """
        v = self.value

        if v.ndim == 2:
            return v

        if v.ndim != 3 or v.shape[-1] < 3:
            raise ValueError("Centroid expects spot as HxW or HxWx3 array.")

        w = self.channelWeight
        # Allow extra channels; only use first 3 as RGB
        I = v[..., 0] * w[0] + v[..., 1] * w[1] + v[..., 2] * w[2]

        return I
