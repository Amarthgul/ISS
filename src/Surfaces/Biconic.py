

from .Surface import *
from Util.Backend import backend as bd
from Util.Backend import constant, backend_name
from Util.PltPlot import DrawAspherical, DrawAsphericalProfile, DrawSphericalProfile
from Util.Globals import ORIGIN, OBJ_FACING, ZERO, ONE, TWO, INFINITY, Axis, SURFACE_COLOR, BOUNDARY_COLOR


class BiconicSurface(Surface):
    def __init__(self,  r, t, sd, m="AIR", K=0, rx = INFINITY, rK = 0):
        """
        Biconic surface. But mostly for modeling cylindrical element.
        """
        super().__init__(r, t, sd, m)


        """Y direction conic factor."""
        self.conic = K

        """X direction radius."""
        self.xRadius = rx

        """X direction conic factor"""
        self.xConic = rK



    def Normal(self, intersections):
        """
        Given the intersections, calculate the normal direction on these intersection points.
        The intersections are treated as on the surface.

        :param intersections: points on the surface.

        :return: Normalized normals of the intersection points on this surface.
        """


    def _CylindricalIntersection(self, incidentRaybatch):
        """
        This method is for when only the radius on one axis has meaningful value and K=0, with the other axis having INF radius. In which case the surface is but cylindrical.
        """
        pass


    def _ConicIntersection(self, incidentRaybatch):
        """
        This method is for when one axis has INF radius and 0 conic (conic does not really matter here), so the surface becomes effectively a 2D sweep instead of a true 3D surface.
        """

        pass


    def _BiconicIntersection(self, incidentRaybatch):
        """
        True biconic intersection solving.

        """
        pass




