
from enum import Enum


from Lens import Lens
from Util.Globals import INFINITY, FAR_DISTANCE


class typeFLE(Enum):
    Vergence = 0
    Linear = 1
    Compensate = 2


class LensFLE(Lens):
    def __init__(self):

        super().__init__()

        """Object distance keys"""
        self.keyObjectDistance = []

        """Surface thicknesses corresponding to the keys"""
        self.keyedSurfaceD = {}


        self.keyedRelation = {}


        """For convenience, many patent data many not start with a 0 index surface, this number marks the index of the first surface in keyedSurfaceD"""
        self.firstSurfaceIndex = 1



    def LoadPresetKeys(self):

        self.keyObjectDistance = [FAR_DISTANCE, 1588, 740]

        self.keyedSurfaceD = {2: [36.814, 26.599, 16.553]}
                             # 5: [6.883, 17.097, 27.144]}

        self.keyedRelation = {2: [typeFLE.Vergence, 0, 0],
                              5: [typeFLE.Compensate, 2, 43.697]}


    def BestFocusBFD(self, distance):

        # Updates the surface thicknesses according to the (object) distance and surface settings

        # Call UpdateLens afterward

        # Then use the same parent method way to calculate best focus BFD
        resolvedThicknesses = {}

        for surfaceIndex in set(self.keyedSurfaceD.keys()) | set(self.keyedRelation.keys()):
            self._ResolvedFLEThickness(surfaceIndex, distance, resolvedThicknesses, set())

        for surfaceIndex, thickness in resolvedThicknesses.items():
            self.surfaces[self._LensSurfaceIndex(surfaceIndex)].thickness = thickness

        self.UpdateLens()

        return super().BestFocusBFD(distance)


    def _ResolvedFLEThickness(self, surfaceIndex, distance, resolvedThicknesses, resolving):
        if surfaceIndex in resolvedThicknesses:
            return resolvedThicknesses[surfaceIndex]

        resolving.add(surfaceIndex)

        relation = self.keyedRelation[surfaceIndex]
        relationType = relation[0]
        if not isinstance(relationType, typeFLE):
            relationType = typeFLE(relationType)

        if relationType == typeFLE.Compensate:
            referenceSurfaceIndex = relation[1]
            compensatedDistance = relation[2]
            thickness = compensatedDistance - self._ResolvedFLEThickness(
                referenceSurfaceIndex,
                distance,
                resolvedThicknesses,
                resolving)
        else:
            if relationType == typeFLE.Vergence:
                thickness = self._InterpolateFLE(distance, self.keyedSurfaceD[surfaceIndex], True)
            elif relationType == typeFLE.Linear:
                thickness = self._InterpolateFLE(distance, self.keyedSurfaceD[surfaceIndex], False)
            else:
                raise ValueError("Unsupported LensFLE relation type " + str(relationType))

        resolving.remove(surfaceIndex)
        resolvedThicknesses[surfaceIndex] = thickness

        return thickness


    def _InterpolateFLE(self, distance, keyedThicknesses, useVergence):
        # if len(self.keyObjectDistance) != len(keyedThicknesses):
        #     raise ValueError("LensFLE object-distance keys and surface-thickness keys must have the same length")

        distance = self._Scalar(distance)

        if useVergence:
            # if distance == 0:
            #     raise ValueError("LensFLE vergence interpolation requires a non-zero object distance")

            x = 1.0 / distance
            keyedX = [1.0 / self._Scalar(d) for d in self.keyObjectDistance]
        else:
            x = distance
            keyedX = [self._Scalar(d) for d in self.keyObjectDistance]

        keyedPairs = sorted(
            (keyedX[i], keyedThicknesses[i])
            for i in range(len(keyedThicknesses)))

        if x <= keyedPairs[0][0]:
            return keyedPairs[0][1]

        if x >= keyedPairs[-1][0]:
            return keyedPairs[-1][1]

        for i in range(len(keyedPairs) - 1):
            x0, y0 = keyedPairs[i]
            x1, y1 = keyedPairs[i + 1]

            if x0 <= x <= x1:
                if x0 == x1:
                    return y0

                return y0 + (y1 - y0) * ((x - x0) / (x1 - x0))

        return keyedPairs[-1][1]


    def _LensSurfaceIndex(self, surfaceIndex):
        lensSurfaceIndex = surfaceIndex - self.firstSurfaceIndex

        # if lensSurfaceIndex < 0 or lensSurfaceIndex >= len(self.surfaces):
        #     raise IndexError("LensFLE surface index " + str(surfaceIndex) + " is out of lens surface range")

        return lensSurfaceIndex


    def _Scalar(self, value):
        if hasattr(value, "get"):
            value = value.get()

        if hasattr(value, "item"):
            value = value.item()

        return float(value)
