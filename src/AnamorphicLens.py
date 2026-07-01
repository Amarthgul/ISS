

from Util.Globals import Axis

from Lens import Lens


class AnamorphicLens(Lens):
    def __init__(self, powerAxis=None):

        super().__init__()

        """This is not a very useful attribute, only old anamorphic lenses have a single power axis. Modern anamorphic all have conic surfaces on both horizontal and vertical axes."""
        self.powerAxis = [Axis.X] if powerAxis is None else list(powerAxis)



