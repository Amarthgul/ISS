


class SurfaceModulator:

    def __init__(self):
        self.frontVertex = None
        self.semiDiameter = None

    def Modulate(self, incidentRB):
        pass


class OnionRingNormal(SurfaceModulator):

    def __init__(self, ringCount=20, disturbance=0.1):
        super().__init__()

        """Total count of ring layers."""
        self.ringCount = ringCount

        """The amount of positional offset that could happen to a ring. 0 being no offset, 1 could make it become tangent with the border of the neighboring ring."""
        self.disturbance = disturbance

        self.normalMap = None


    def GenerateNormalMap(self):
        pass


    def Visualize(self):
        pass


class Dust(SurfaceModulator):

    def __init__(self, maxDust=10):
        super().__init__()

        self.maxDust = maxDust

