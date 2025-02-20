
"""
This module records the path of the rays so that some debugging and inpection features are eaiser to perform. It is also used in establishing the parameters of the lens, such as tracing the focal point, principle point, and entrance pupil. 

It is not recommended to use this module in the ray tracing process of production imagings as it may signicantly increase the computational cost and memory usage.

"""

from Util.Backend import backend as bd
from Util.Globals import ZERO, NEAR_ZERO, OBJ_FACING, SOME_BIG_CONST, AXIAL_ZERO, Axis
from Util.Misc import ArrayMagnitude, Magnitude, TransversalDistance
from Util.PltPlot import DrawLines, DrawNormal



class RayPath():
    def __init__(self):
        self.position = []
        self.direction = []
        self.reflected = []
        self.vignetted = []


    def Append(self, raybatch, reflected = None, vignetted = None):
        """
        Append a raybatch to the path. 

        :param raybatch: The raybatch to be recorded.
        :param reflected: bool array of reflected rays.
        :param vignetted: bool array of vignetted rays.
        """

        if (self.position is None):
            # Position and direction should come in pairs and thus are recorded together.
            self.position = raybatch.Position()
            self.direction = raybatch.Direction()
        else:
            self.position.append(raybatch.Position())
            self.direction.append(raybatch.Direction())

        if (self.reflected is None):
            self.reflected = bd.ones_like(self.position).astype(bd.bool_)
        else:
            self.reflected.append(reflected)

        if(self.vignetted is None):
            self.vignetted = bd.ones_like(self.position).astype(bd.bool_)
        else:
            self.vignetted.append(vignetted)


    def DrawPath(self, expendEnd = 0.0, omitIncident = True, color = 'red'):
        """
        Draw the path of the recorded rays. 

        :param expendEnd: The length of the path after the last surface.
        :param omitIncident: when enabled, the fist set of rays will not be plotted, in most cases this refers to the incident ray. 
        :param color: color of the oath lines. 
        """
        
        if(len(self.position) <= 1):
            raise ValueError("The path is too short to plot.")

        start = 0 
        if(omitIncident):
            start = 1 
        end = len(self.position) - 1

        for i in range(start, end):
            # Bool filter out the rays that are vignetted or reflected 
            # so that both set of points have the same size 
            if(len(self.position[i+1]) == 0): break 
            DrawLines(
                self.position[i][~self.vignetted[i+1]][~self.reflected[i+1]],    
                self.position[i+1], 
                lineColor = color, 
                lineWidth = 0.5
            )

        # Draw the last point
        if(expendEnd > 0):
            i = len(self.position)-1
            DrawNormal(
                self.position[i][~self.reflected[i]], 
                self.direction[i][~self.reflected[i]], 
                lineColor = color,
                lineLength = expendEnd
                )


    def ExitingPairs(self, invertDirection = True):
        """
        Get the exiting ray pairs.
        """
        if(invertDirection):
            return self.position[-1], -self.direction[-1]
        return self.position[-1], self.direction[-1]


    def FindOutermost(self, position, direction):
        """
        Find the outermost rays that are not vignetted. 
        """
        

        pass 


    def FindConvergingPoint(self, position, direction, enablePrune=True):
        """
        Find the converging point of the exiting rays. 

        :param position: array of ray positions on surfaces.
        :param direction: array of ray directions corresponding to the positions, must be normalized.

        :return: a single point towards which the rays are converging.
        """

        # The following genius code are provided by 
        # https://stackoverflow.com/users/654602/tyler-fox 

        da = (direction * position).sum(axis=-1, keepdims=True)
        b = (direction * da - position).sum(axis=0)
        c, d = position.shape
        m = bd.inner(direction.T, direction.T) - bd.diag(bd.full(d, c))

        # This calculation will return non zero for on axis results, 
        # thus needing to replace small numbers with 0 
        solved = bd.linalg.solve(m, b) 


        # Oblique rays will disrupt the convergence, so some prone may be used 
        if(enablePrune):
            solved = self._ProneOffAxisZ(solved, position, direction)

        # Replace the near zeros with zeros 
        solved = bd.where(bd.abs(solved) < NEAR_ZERO, ZERO, solved)

        return solved

        
    def FindAxialIntersection(self, position, direction):
        """
        Find the intersection between a ray and the optical axis (where x=y=0). 

        :param position: array of ray positions on surfaces.
        :param direction: array of ray directions corresponding to the positions.

        :return: array point(s) of intersection, the array size is the same as input.
        """

        pass


    def DepthIntersect(self, zDepthPoint):
        """
        Try to find the intersections of entire raypath at given z depth. 

        :param zDepthPoint: point whose z location will be used for calculation. 

        :return: set of verteices what interset at the depth, if any. 
        """

        posCount = len(self.position)
        intersections = bd.array([])

        # Since a plane can cover several surfaces, iterate through all the segments 
        for i in range(posCount):
            if(i < posCount-1):
                thisPos = self.position[i][~self.vignetted[i+1]]
                nextPos = self.position[i+1]
                temp = self._LineSegmentIntersection(
                        thisPos, nextPos, 
                        zDepthPoint, OBJ_FACING
                    )

                # For non intersects this returns Nan array, check for not Nan entries
                naNan = temp[~bd.any(bd.isnan(temp), axis=1)]

                # Concatenate array if there are non Nan entries 
                if(len(naNan) > 0):
                    if(len(intersections)==0):
                        intersections = naNan.copy()
                    else: 
                        intersections = bd.concatenate((intersections, naNan))

        return intersections


    def PruneAll(self):
        """
        Prune all the rays that are vignetted or reflected, leaving only the refeacted rays. Not that this only works for refractive optics. 
        """
        doppelganger = RayPath()
        doppelganger.position = list.copy(self.position)
        doppelganger.direction = list.copy(self.direction)
        doppelganger.reflected = list.copy(self.reflected)
        doppelganger.vignetted = list.copy(self.vignetted)

        ineffective = doppelganger.vignetted[len(doppelganger.position)-1] 
        ineffective[~ineffective] &= doppelganger.reflected[len(doppelganger.position)-1]

        for i in range(len(doppelganger.position)-1, -1, -1):
            if(i < len(doppelganger.position)-1 and i >1):
                # Bool AND the ineffective rays 
                doppelganger.vignetted[i-1][~doppelganger.vignetted[i-1]] |= doppelganger.vignetted[i]
                doppelganger.reflected[i-1][~doppelganger.vignetted[i]] |= doppelganger.reflected[i][~doppelganger.vignetted[i+1]]

                # Merge the reflected and vignetted 
                ineffective = doppelganger.vignetted[i] 
                ineffective[~doppelganger.vignetted[i]] |= doppelganger.reflected[i][~doppelganger.vignetted[i+1]]

                # Apply the ineffective rays to prune the positions  and directions 
                doppelganger.position[i-1] = doppelganger.position[i-1][~ineffective]
                doppelganger.direction[i-1] = doppelganger.direction[i-1][~ineffective]
        
        # In the event of the raypath containing the emission source, index 0 is None
        if(doppelganger.vignetted[0] == None):
            # In this case, the vignetted mask of the emission needs to be manually applied from the first entry
            temp = doppelganger.vignetted[1]
            temp[~temp] = ineffective[~ineffective]
            ineffective = temp
            doppelganger.position[0] = doppelganger.position[0][~ineffective]
            doppelganger.direction[0] = doppelganger.direction[0][~ineffective]

        # Since ray positions and directions are all pruned, reset all vignetted and relfected masks to false. 
        for i in range(len(doppelganger.position)-1, 0, -1):
            doppelganger.vignetted[i] = bd.zeros(len(doppelganger.position[i])).astype(bd.bool_)
            doppelganger.reflected[i] = bd.zeros(len(doppelganger.position[i])).astype(bd.bool_)

        return doppelganger


    def EndToEndIntersection(self):
        """
        Find the intersections of the incident rays and the exiting rays. 
        This assume the path include the light source and is infinite conjugate. 
        """

        P1 = self.position[0]
        D1 = self.direction[0]
        P2 = self.position[len(self.position)-1]
        D2 = self.direction[len(self.direction)-1]

        # There might be ray vecs that's on the optical axis at the beginning,
        # and for axial symmetric lens it'll stay on the axis through to the end. 
        # This will make later linalg.solve to fail when using numpy. 
        # the following steps are to check for on axis rays and remove them. 
        bP1 = bd.isclose(TransversalDistance(P1), ZERO, AXIAL_ZERO) 
        bD1 = bd.isclose(TransversalDistance(D1), ZERO, AXIAL_ZERO) 
        bP2 = bd.isclose(TransversalDistance(P2), ZERO, AXIAL_ZERO) 
        bD2 = bd.isclose(TransversalDistance(D2), ZERO, AXIAL_ZERO) 
        onAxis = (bP1 & bD1 & bP2 & bD2)    
        P1 = P1[~onAxis[:, 0]]
        D1 = D1[~onAxis[:, 0]]
        P2 = P2[~onAxis[:, 0]]
        D2 = D2[~onAxis[:, 0]]

        # Clip them to use only the YZ coordinates 
        A = bd.stack([D1[:, 1:], -D2[:, 1:]], axis=2) 
        B = P2[:, 1:] - P1[:, 1:] 

        t = bd.linalg.solve(A, B)
        intersections = P1 + t[:, 0, bd.newaxis] * D1  # Shape (n, 3)

        return intersections


    # ==================================================================
    """ ====================== Private Methods ===================== """
    # ==================================================================


    def _zPlaneIntersections(self, zDepth, positions, directions):
        """
        Calculate the intersections between rays (vectors from points) and a 3D plane in square shape.
        :param surfaceIndex: the index of the surface to intersect. 
        """

        # TODO: add tilt shift support here
        imager_normal = bd.array([0, 0, -1])
        planePoint = bd.array([ZERO, ZERO, zDepth])
        
        # Calculate d (the offset from the origin in the plane equation ax + by + cz + d = 0)
        d = -bd.dot(imager_normal, planePoint)

        # Calculate dot product of direction vectors with the plane normal
        denom = bd.dot(directions, imager_normal)
        
        # Avoid division by zero (for parallel rays)
        # valid_rays = (denom != 0)

        # For valid rays, calculate t where the intersection occurs
        t = -(bd.dot(positions, imager_normal) + d) / denom
        
        # Calculate the intersection points
        intersections = positions + t[:, bd.newaxis] * directions

        return intersections  


    def _LineSegmentIntersection(self, P1, P2, planePoint, planeNormal):
        """
        Given set of lines defined by two ends P1 and P2, find if there are any intersections between the lines and a plane. 

        :param P1: Position of one end of the line.
        :param P2: Position of the other end of line. 
        :param planePoint: A point on the plane. 
        :param planeNormal: the normal of the plane. 

        :return: array of points of intersection, Nan entries if no intersection. 
        """
        # Calculate direction vectors for all line segments
        line_direction = P2 - P1
        
        # Calculate the dot product for numerator and denominator (vectorized)
        numerator = bd.dot((planePoint - P1), planeNormal)
        denominator = bd.dot(line_direction, planeNormal)
        
        # Handle parallel lines (denominator close to zero)
        parallelMask = bd.abs(denominator) < NEAR_ZERO
        t = numerator[~parallelMask] / denominator[~parallelMask]
        
        # Calculate intersection points
        intersection_points = P1 + t[:, bd.newaxis] * line_direction

        # Mask out invalid intersections
        intersection_points[(t < 0) | (t > 1) | parallelMask] = bd.nan
        
        return intersection_points


    def _ProneOffAxisZ(self, intersection, positions, directions, threshold = 0.1):
        """
        Prone converging rays whoese position is too far away from the given intersection until the Z axis delta is smaller than the threshold. 

        :param intersection: single point of convergence. 
        :param positions: list of ray positions. 
        :param directions: list of ray directions. 
        :param threshold: scalar threshold to judge the delta change. 

        :return: a theoretically more accurate point.  
        """

        delta = SOME_BIG_CONST

        while(len(positions)>2 and delta>threshold):
            dist = ArrayMagnitude((positions - intersection)[:, :2])
            largeInd = bd.argmax(dist)
            positions = bd.concatenate([positions[:largeInd], positions[largeInd + 1:]])
            directions = bd.concatenate([directions[:largeInd], directions[largeInd + 1:]])
            newIntersect = self.FindConvergingPoint(positions, directions, False)

            # Only testing the z axis delta 
            delta = Magnitude((newIntersect - intersection)[2])
            intersection = newIntersect

        return intersection








def main():
    pass 


if __name__ == "__main__":
    main()

