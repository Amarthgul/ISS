



from .Surface import *
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile, DrawSphericalProfile
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR


class DonutSpherical(Surface):
    def __init__(self, r, t, sd, m, iR):
        """This is basically the same as standard spherical, but there is a hole in the middle. """
        super().__init__(r, t, sd, m)

        self.innerRadius = iR


    def Intersection(self, incidentRaybatch):
        """
        Given a raybatch, calculate the intersection of these rays on this surface and return the intersection coordinates.

        :param incidentRaybatch: RayBatch that will be tested for intersection.

        :return: An array of intersections, a bull secondary array, the bool array of vignetted.
        """

        if(self.radius == INFINITY):
            initialIntersection = self._PlaneIntersection(incidentRaybatch)
        else:
            initialIntersection = self._SphericalIntersection(incidentRaybatch)

        # TODO: exclude the ones that are in innerRadius, then return the results

