


from .ClearBoundary import ClearBoundary

from Material import Material
from Raytracing.RayBatch import RayBatch, GenerateBeam
from Raytracing.Reflection import Reflect
from Raytracing.Refraction import Refract
from Raytracing.Polarization import SenkrechtUndParallel, PolarizeRB, ResidueRB, FresnelReflectance, QuantitativePolarize
from Util.Backend import backend as bd
from Util.Backend import constant
from Util.MathFunctions import NewtonSolver
from Util.Misc import ArrayNormalized, TransversalDistance
from Util.Globals import OBJ_FACING, Axis, RNG, PBR
from Util.SpatialEllipse import SpatialEllipse, SpatialCircle
from Util.PltPlot import DrawSpherical, DrawPoints, DrawDirection, DrawNormal, DrawRaybatch, SetUnifScale, RemoveBG, AddXYZ, DrawEllipse, DrawClearBoundary



class MetalBoundary(ClearBoundary):
    def __inti__(self, e1, e2, mat=PBR.METAL):
        super().__init__(e1, e2, mat=mat)

        """Specific attribute for metallic surface. This value controls how much of the incident rays will be absorbed and no longer participate in the ray transfer. """
        self.absorption = 0.1


    def DrawSurface(self):
        DrawClearBoundary(self.E1, self.E2, surfaceColor='r')


    def Trace(self, incidentRaybatch, previousRI=None, inverted=False):
        """
        A clear boundary still calculates refraction, but it is only for Frensnel reflectance and not for ray propagation.

        :return: reflected raybatch and a boolean mask indicating the rays that did intersect.
        """

        #TODO: add reflectance support

        intersections, _mask = self.Intersection(incidentRaybatch)

        # Reflected RB is created here first with the mask applied, this means it should not contain any rays that are not intersecting with the surface.
        reflectedRB = RayBatch(bd.copy(incidentRaybatch.value[_mask]))
        reflectedRB.SetPosition(intersections)

        # Becasue reflected
        directions = reflectedRB.Direction()

        # Calculate the normal direction
        normals = self.Normal(intersections)
        # Accquire a desired normal vector direction as they should be pointing against the incident rays
        desiredDirection = -bd.sign(reflectedRB.Direction()[:, 2])
        # Only flip the normals if the clear boundary is not a cylinder
        if (self.E1.SemiAxisMagnititude() != self.E2.SemiAxisMagnititude()):
            normals[desiredDirection != bd.sign(normals[:, 2])] *= -1

        # Add some jittering to the normal direction to approximate diffuse reflection
        randomDirection = self._RandomInHemisphere(normals)
        normals = ArrayNormalized(normals * self.specularReflection + \
                                  randomDirection * (1 - self.specularReflection))
        # Note that this is not the Lambertian reflected radiant intensity as viewing angle is not considered

        # Calculate the reflection directions and directly update the reflected RB
        reflected = Reflect(reflectedRB.Direction(), normals)
        reflectedRB.SetDirection(reflected)

        reflectedRB.RandomDrop(1 - self.absorption)

        return reflectedRB, _mask




