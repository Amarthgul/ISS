"""Small Taylor polynomials and Gaussian references for aberration analysis.

Variables are normalized sagittal pupil x, tangential pupil y, and field h.
All algebra is CPU float64, truncated by total degree, not a sampled fit.
The centered reference uses a paraxial entrance pupil (no nonlinear stop aiming).
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from Util.Analysis.Paraxial import ParaxialResult


@lru_cache(maxsize=4)
def _Basis(degree):
    powers = tuple((x, y, total - x - y)
                   for total in range(degree + 1)
                   for x in range(total + 1)
                   for y in range(total - x + 1))
    indices = {power: index for index, power in enumerate(powers)}
    left, right, output = [], [], []
    for i, a in enumerate(powers):
        for j, b in enumerate(powers):
            power = tuple(a[k] + b[k] for k in range(3))
            if sum(power) <= degree:
                left.append(i)
                right.append(j)
                output.append(indices[power])
    return (powers, indices, np.array(left), np.array(right), np.array(output))


class Taylor:
    """Dense three-variable polynomial; binary operands share the same degree.

    Use Scale/AddConstant for scalar operations. Missing monomials are zero.
    Power accepts real exponents when the constant term is positive.
    """

    def __init__(self, coefficients, degree):
        self.coefficients = coefficients
        self.degree = degree

    @classmethod
    def Constant(cls, value, degree):
        coefficients = np.zeros(len(_Basis(degree)[0]))
        coefficients[0] = value
        return cls(coefficients, degree)

    @classmethod
    def Variable(cls, axis, degree):
        result = cls.Constant(0.0, degree)
        power = tuple(int(i == axis) for i in range(3))
        result.coefficients[_Basis(degree)[1][power]] = 1.0
        return result

    def __add__(self, other):
        return Taylor(self.coefficients + other.coefficients, self.degree)

    def __sub__(self, other):
        return Taylor(self.coefficients - other.coefficients, self.degree)

    def __mul__(self, other):
        powers, _indices, left, right, output = _Basis(self.degree)
        coefficients = np.bincount(
            output, weights=self.coefficients[left] * other.coefficients[right],
            minlength=len(powers))
        return Taylor(coefficients, self.degree)

    def Scale(self, value):
        return Taylor(self.coefficients * value, self.degree)

    def AddConstant(self, value):
        coefficients = self.coefficients.copy()
        coefficients[0] += value
        return Taylor(coefficients, self.degree)

    def Power(self, exponent):
        constant = self.coefficients[0]
        delta = self.Scale(1.0 / constant).AddConstant(-1.0)
        term = Taylor.Constant(1.0, self.degree)
        result = term
        for k in range(1, self.degree + 1):
            term = (term * delta).Scale((exponent - k + 1) / k)
            result = result + term
        return result.Scale(constant ** exponent)

    def Select(self, degrees):
        coefficients = self.coefficients.copy()
        for index, power in enumerate(_Basis(self.degree)[0]):
            if sum(power) not in degrees:
                coefficients[index] = 0.0
        return Taylor(coefficients, self.degree)

    def AsDict(self):
        """Map (x power, y power, field power) to nonzero coefficients."""
        return {power: float(value) for power, value in
                zip(_Basis(self.degree)[0], self.coefficients) if value != 0.0}

    def Evaluate(self, x, y, h):
        return math.fsum(value * x**a * y**b * h**c
                         for (a, b, c), value in self.AsDict().items())


@dataclass(frozen=True)
class WavefrontCoefficients:
    """Wijk multiplies h^i rho^j cos(phi)^k, in mm (not waves).

    phi=0 is the tangential +y pupil direction. Field-only piston and
    degrees below four are omitted. The reference is a Gaussian sphere/plane,
    with positive W meaning reference optical path minus actual optical path.
    """
    values: dict[str, float]

    def AsDict(self):
        return dict(self.values)

    def ForOrder(self, order):
        return WavefrontCoefficients({key: value for key, value in self.values.items()
                                      if int(key[1]) + int(key[2]) == order + 1})

    def Evaluate(self, rho, phi, h=1.0):
        return math.fsum(value * h**int(key[1]) * rho**int(key[2]) *
                         math.cos(phi)**int(key[3])
                         for key, value in self.values.items())


def WavefrontFromTaylor(polynomial):
    """Resolve a rotationally symmetric polynomial into the Wijk basis."""
    _powers, indices, *_ = _Basis(polynomial.degree)
    residual = polynomial.coefficients.copy()
    values = {}
    for degree in range(4, polynomial.degree + 1, 2):
        for i in range(degree):
            j = degree - i
            for k in range(i % 2, min(i, j) + 1, 2):
                m = (j - k) // 2
                value = residual[indices[(2 * m, k, i)]]
                values[f"W{i}{j}{k}"] = float(value)
                for b in range(m + 1):
                    residual[indices[(2 * (m - b), 2 * b + k, i)]] -= (
                        value * math.comb(m, b))
    return WavefrontCoefficients(values)


@dataclass(frozen=True)
class GaussianReference:
    """Reference at a vertex: finite Gaussian sphere or collimated plane.

    imageDistance is measured from this vertex. imageHeight is at h=1.
    A finite image must be distinct from this vertex. When the marginal
    slope is zero, chiefSlope specifies the normal of the reference plane.
    """
    imageDistance: float
    imageHeight: float
    refractiveIndex: float
    chiefSlope: float

    @classmethod
    def FromSurface(cls, trace):
        marginal, chief = trace.marginalOutgoing, trace.chiefOutgoing
        if marginal.slope == 0.0:
            return cls(math.inf, 0.0, trace.outgoingRI, chief.slope)
        distance = -marginal.height / marginal.slope
        return cls(distance, chief.height + distance * chief.slope,
                   trace.outgoingRI, chief.slope)

    def Wavefront(self, x, y, u, v, path):
        """Close accumulated OPL on the Gaussian sphere/plane through the vertex.

        Use the ray/sphere intersection, including the higher-order direction
        and pupil geometry. The collimated branch intersects the plane with
        normal (0, h*chiefSlope, 1). Pupil-independent phase is omitted when
        resolving Wijk. No paraxial conversion of transverse ray errors is used.
        """
        h = Taylor.Variable(2, x.degree)
        normSquared = (u * u + v * v).AddConstant(1.0)
        if math.isinf(self.imageDistance):
            slope = h.Scale(self.chiefSlope)
            travel = (y * slope * normSquared.Power(0.5) *
                      (v * slope).AddConstant(1.0).Power(-1.0)).Scale(-1.0)
        else:
            centerY = h.Scale(self.imageHeight)
            distance = self.imageDistance
            dot = (x * u + (y - centerY) * v).AddConstant(-distance) * normSquared.Power(-0.5)
            offset = x * x + y * y - (y * centerY).Scale(2.0)
            root = (dot * dot - offset).Power(0.5)
            travel = dot.Scale(-1.0) - root.Scale(math.copysign(1.0, distance))
        return WavefrontFromTaylor((path + travel.Scale(self.refractiveIndex)).Scale(-1.0))


@dataclass(frozen=True)
class RayReference:
    """Full-system Gaussian image and common paraxial input pupil.

    paraxial is a ParaxialResult. The image is finite and its distance from
    the last vertex is nonzero. Distances/OPL are mm; wavelength is nm.
    Angular fields retain TraceLens's slope convention: h*radians(field)
    is dy/dz, so the physical ray angle is atan of that slope.
    """
    paraxial: ParaxialResult
    referenceField: float
    fieldUnits: str

    @property
    def imageDistance(self):
        return self.paraxial.ImageDistance()

    @property
    def imageHeight(self):
        chief = self.paraxial.surfaces[-1].chiefOutgoing
        return chief.height + self.imageDistance * chief.slope

    @property
    def imageReference(self):
        last = self.paraxial.surfaces[-1]
        return GaussianReference(self.imageDistance, self.imageHeight,
                                 last.outgoingRI, last.chiefOutgoing.slope)

    def Launch(self, degree):
        """Return vertex-plane x,y, slopes dx/dz,dy/dz and incident OPL."""
        px, py, h = (Taylor.Variable(axis, degree) for axis in range(3))
        first = self.paraxial.surfaces[0]
        marginal, chief = first.marginalIncident, first.chiefIncident
        x = px.Scale(marginal.height)
        y = py.Scale(marginal.height) + h.Scale(chief.height)
        u = px.Scale(marginal.slope)
        v = py.Scale(marginal.slope) + h.Scale(chief.slope)
        normSquared = (u * u + v * v).AddConstant(1.0)
        if math.isinf(self.paraxial.objectDistance):
            path = ((x * u + y * v) * normSquared.Power(-0.5)).Scale(first.incidentRI)
        else:
            path = normSquared.Power(0.5).AddConstant(-1.0).Scale(
                first.incidentRI * self.paraxial.objectDistance)
        return x, y, u, v, path

    def Wavefront(self, x, y, u, v, path):
        """OPD on the complete system's Gaussian reference sphere."""
        return self.imageReference.Wavefront(x, y, u, v, path)
