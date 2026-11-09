
"""
First-order optics in mm and radians; field inputs are in degrees.

Indices and axial distances are signed for reflected paths. Pupil positions
are relative to the first vertex. Object distance is positive to its left.
Call Update after prescription edits. A system without a marked stop uses
stopIndex (initially zero), with unit semi-diameter if none is specified.
"""

from enum import Enum
from dataclasses import dataclass
import csv
import math

import numpy as np

from Util.Analysis.Paraxial import (
    IDENTITY, MatMul, RefractionMatrix, TranslationMatrix,
)


def _Divide(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.divide(numerator, denominator))


@dataclass(frozen=True)
class YNURow:
    surfaceIndex: int
    z: float
    y: float
    incidentN: float
    incidentU: float
    n: float
    u: float
    thickness: float = 0.0

    @property
    def nu(self):
        return self.n * self.u

    @property
    def incidentNU(self):
        return self.incidentN * self.incidentU


@dataclass(frozen=True)
class Pupil:
    position: float
    semiDiameter: float


@dataclass(frozen=True)
class YNUTrace:
    marginal: tuple[YNURow, ...]
    chief: tuple[YNURow, ...]

    normalizedMarginal: tuple[YNURow, ...] = None
    normalizedChief: tuple[YNURow, ...] = None

    def getYNU(self, filename="YNU.csv"):
        columns = ["Object"] + [f"Element {row.surfaceIndex}" for row in self.marginal]
        rows = []

        def add(name, values):
            rows.append(dict(quantity=name, **dict(zip(columns, values))))

        add("thickness", [0.0] + [row.thickness for row in self.marginal])
        add("n", [self.marginal[0].incidentN] + [row.n for row in self.marginal])
        rays = [("marginal", self.marginal), ("chief", self.chief)]
        if self.normalizedMarginal is not None:
            rays.extend([("normalized_marginal", self.normalizedMarginal),
                         ("normalized_chief", self.normalizedChief)])
        for name, ray in rays:
            add(name + "_y", [ray[0].y] + [row.y for row in ray])
            add(name + "_u", [ray[0].incidentU] + [row.u for row in ray])
            add(name + "_nu", [ray[0].incidentNU] + [row.nu for row in ray])
        if filename is not None:
            with open(filename, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return rows


class ParaxialType(Enum):
    Standard = 0  # Refractive surface
    Mirror = 1
    Stop = 2


class LiteElement:
    """
    Either a lens or a surface in 1st order approximation.
    """
    def __init__(self, r=None, c=None, t=None, f=None, sd=None, fRI=1, RI=1):
        self.type = ParaxialType.Standard
        self.isStop = False
        self._curvature = c if c is not None else (
            _Divide(1.0, r) if r is not None else 0.0)
        self._power = None if f is None else _Divide(RI, f)
        self.thickness = 0.0 if t is None else t # Distance to the next element
        self.clearSemiDiameter = sd
        self.fRI = fRI # Refractive index in front of this element
        self.RI = RI   # Refractive index after this element


    @property
    def radius(self):
        return _Divide(1.0, self.curvature)

    @radius.setter
    def radius(self, value):
        self.curvature = _Divide(1.0, value)

    @property
    def curvature(self):
        return self._curvature

    @curvature.setter
    def curvature(self, value):
        self._curvature = value
        self._power = None

    @property
    def phi(self):
        if self.type == ParaxialType.Stop:
            return 0.0
        if self._power is not None:
            return self._power
        return (self.RI - self.fRI) * self.curvature

    @phi.setter
    def phi(self, value):
        # Explicit power also represents a thin lens in a single medium.
        self._power = value


    def MarkAsStop(self, sd):
        self.type = ParaxialType.Stop
        self.clearSemiDiameter = sd


    def MarkAsMirror(self):
        self.type = ParaxialType.Mirror

    def Update(self):
        if self.type == ParaxialType.Stop:
            self.RI = self.fRI
        elif self.type == ParaxialType.Mirror:
            self.RI = -self.fRI
        self.matrix = RefractionMatrix(self.phi, self.fRI, self.RI)
        return self


class LiteStop(LiteElement):
    def __init__(self, r=None, c=None, t=None, f=None, sd=None, fRI=1, RI=1):
        super().__init__(r, c, t, f, sd, fRI, RI)

        self.isStop = True
        if r is None and c is None and f is None:
            self.type = ParaxialType.Stop


class LiteSystem:
    """
    An optical system consisting of 1st order elements.
    """
    def __init__(self, writeMarkdown=False):
        self.elements = []
        self.writeMarkdown = writeMarkdown

        self.referenceField = None
        self.stopIndex = 0

        # Calculated system parameters
        self.EFL = None
        self.BFD = None

        self.EP = None # Entrance pupil position and size
        self.XP = None # Exit pupil position and size


        # Infinite conjugate rays

        self.marginal = None # Tracing with normalized height/slope
        self.chief = None

        self.marginalScaled = None # Height scaled to the stop
        self.chiefScaled = None

        self.unVignettedSizes = None
        self.imageDistance = None
        self.imageHeight = None
        self.systemData = None


    def _Matrix(self, start, end):
        matrix = IDENTITY
        for index in range(start, end):
            element = self.elements[index]
            matrix = MatMul(element.matrix, matrix)
            if index < end - 1:
                matrix = MatMul(TranslationMatrix(element.thickness), matrix)
        return matrix


    def Update(self):
        previousRI = self.elements[0].fRI
        z = 0.0
        for element in self.elements:
            element.fRI = previousRI
            element.Update()
            element.vertexZ = z
            previousRI = element.RI
            z += element.thickness

        self.stopIndex = next((i for i, element in enumerate(self.elements)
                               if element.isStop or element.type == ParaxialType.Stop), self.stopIndex)
        stop = self.elements[self.stopIndex]
        self.stopSemiDiameter = (1.0 if stop.clearSemiDiameter is None
                                 else stop.clearSemiDiameter)
        self.systemMatrix = self._Matrix(0, len(self.elements))
        a, b = self.systemMatrix[0]
        c, d = self.systemMatrix[1]
        self.EFL = _Divide(-1.0, c)
        self.BFD = _Divide(-a, c)

        self.stopMatrix = self._Matrix(0, self.stopIndex + 1)
        sa, sb = self.stopMatrix[0]
        self.EP = Pupil(_Divide(sb, sa), _Divide(self.stopSemiDiameter, abs(sa)))

        exitMatrix = self._Matrix(self.stopIndex, len(self.elements))
        ea, eb = exitMatrix[0]
        ec, ed = exitMatrix[1]
        self.XP = Pupil(self.elements[-1].vertexZ + _Divide(-eb, ed),
                        abs(_Divide((ea * ed - eb * ec) * self.stopSemiDiameter, ed)))

        self.marginal = self.TraceRay(1.0, 0.0)
        self.chief = self.TraceRay(-self.EP.position, 1.0)
        self.marginalScaled = self.TraceRay(self.stopSemiDiameter / sa, 0.0)
        self.TraceChief(0.0 if self.referenceField is None else self.referenceField)
        return self


    def TraceRay(self, y, u=0.0, nu=None):
        if nu is not None:
            u = nu / self.elements[0].fRI
        rows = []
        for index, element in enumerate(self.elements):
            outgoingU = element.matrix[1][0] * y + element.matrix[1][1] * u
            rows.append(YNURow(index, element.vertexZ, y, element.fRI, u,
                               element.RI, outgoingU, element.thickness))
            u = outgoingU
            if index < len(self.elements) - 1:
                y += element.thickness * u
        return tuple(rows)


    def TraceChief(self, halfAngleField):
        slope = math.radians(halfAngleField)
        self.chiefScaled = self.TraceRay(-self.EP.position * slope, slope)
        self._SetTraceData(YNUTrace(self.marginalScaled, self.chiefScaled))
        self.GetSystemData()
        return self.chiefScaled


    def _SetTraceData(self, trace):
        # Minimum centered clear semi-diameters for the full field bundle.
        self.unVignettedSizes = [abs(marginal.y) + abs(chief.y)
                                 for marginal, chief in zip(trace.marginal, trace.chief)]
        marginal, chief = trace.marginal[-1], trace.chief[-1]
        self.imageDistance = _Divide(-marginal.y, marginal.u)
        self.imageHeight = chief.y + self.imageDistance * chief.u


    def _FiniteConjugate(self, objectHeight, objectDistance):
        a, b = self.stopMatrix[0]
        marginalSlope = self.stopSemiDiameter / (a * objectDistance + b)
        chiefSlope = -a * objectHeight / (a * objectDistance + b)
        trace = YNUTrace(
            self.TraceRay(objectDistance * marginalSlope, marginalSlope),
            self.TraceRay(objectHeight + objectDistance * chiefSlope, chiefSlope),
            self.TraceRay(objectDistance, 1.0),
            self.TraceRay(-self.EP.position, 1.0))
        self._SetTraceData(trace)
        self.GetSystemData()
        return trace


    def FiniteConjugateAngle(self, halfAngleField, objectDistance):
        height = -(objectDistance + self.EP.position) * math.radians(halfAngleField)
        return self._FiniteConjugate(height, objectDistance)


    def FiniteConjugateHeight(self, objectHeight, objectDistance):
        return self._FiniteConjugate(objectHeight, objectDistance)


    def getYNU(self, filename="YNU.csv", scaled=True):
        trace = (YNUTrace(self.marginalScaled, self.chiefScaled, self.marginal, self.chief) if scaled
                 else YNUTrace(self.marginal, self.chief, self.marginal, self.chief))
        rows = trace.getYNU(filename)
        # Normalized table output does not change the physical aperture/field.
        self._SetTraceData(YNUTrace(self.marginalScaled, self.chiefScaled))
        self.GetSystemData()
        return rows


    def GetNYU(self, filename="YNU.csv", scaled=True):
        return self.getYNU(filename, scaled)


    def GetYNU(self, filename="YNU.csv", scaled=True):
        return self.getYNU(filename, scaled)


    def GetSystemData(self):
        lines = [
            "Paraxial system prescription (mm; RI dimensionless)",
            "Element\tRadius\tThickness\tRI\tUnvignetted semi-diameter",
        ]
        for index, (element, size) in enumerate(zip(self.elements, self.unVignettedSizes)):
            lines.append(f"{index}\t{element.radius:.10g}\t{element.thickness:.10g}"
                         f"\t{element.RI:.10g}\t{size:.10g}")
        lines.extend([
            f"EFL: {self.EFL:.10g} mm",
            f"BFD (infinite conjugate): {self.BFD:.10g} mm",
            f"Image distance (from last vertex): {self.imageDistance:.10g} mm",
            f"Image height: {self.imageHeight:.10g} mm",
            f"EP: position={self.EP.position:.10g} mm, semi-diameter={self.EP.semiDiameter:.10g} mm",
            f"XP: position={self.XP.position:.10g} mm, semi-diameter={self.XP.semiDiameter:.10g} mm",
            "Pupil positions are relative to the first vertex.",
        ])
        self.systemData = "\n".join(lines)
        if self.writeMarkdown:
            with open("LiteSurface.md", "w", encoding="utf-8") as stream:
                stream.write("# Lite system report\n\n```text\n" + self.systemData + "\n```\n")
        return self.systemData
