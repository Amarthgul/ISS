
"""
This module records the path of the rays so that some debugging and inpection features are eaiser to perform. 

It is not recommended to use this module in the ray tracing process of production imagings as it may signicantly increase the computational cost and memory usage.

"""

from Util.Backend import backend as bd
from Util.PlotTest import DrawLines, DrawNormal



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


    def PlotPath(self, expendEnd = 0.0):
        """
        Draw the path of the recorded rays. 

        :param expendEnd: The length of the path after the last surface.
        """
        
        if(len(self.position) <= 1):
            raise ValueError("The path is too short to plot.")

        for i in range(len(self.position) - 1):
            # Bool filter out the rays that are vignetted or reflected 
            # so that both set of points have the same size 
            DrawLines(
                self.position[i][~self.vignetted[i+1]][~self.reflected[i+1]],    
                self.position[i+1], 
                lineColor = 'red', 
                lineWidth = 0.5
            )

        # Draw the last point
        if(expendEnd > 0):
            i = len(self.position)-1
            DrawNormal(
                self.position[i][~self.reflected[i]], 
                self.direction[i][~self.reflected[i]], 
                lineColor = 'red',
                lineLength = expendEnd
                )


    def ExitingPairs(self, invertDirection = False):
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


    def FindConvergingPoint(self, position, direction):
        """
        Find the converging point of the exiting rays. 

        :param position: array of ray positions on surfaces.
        :param direction: array of ray directions corresponding to the positions.

        :return: a single point towards which the rays are converging.
        """

        # The following genius code are provided by 
        # https://stackoverflow.com/users/654602/tyler-fox 

        da = (direction * position).sum(axis=-1, keepdims=True)
        b = (direction * da - position).sum(axis=0)
        c, d = position.shape
        m = bd.inner(direction.T, direction.T) - bd.diag(bd.full(d, c))

        return bd.linalg.solve(m, b)
        


    def FindAxialIntersection(self, position, direction):
        """
        Find the intersection between a ray and the optical axis (where x=y=0). 

        :param position: array of ray positions on surfaces.
        :param direction: array of ray directions corresponding to the positions.

        :return: array point(s) of intersection, the array size is the same as input.
        """

        pass
















def main():
    pass 


if __name__ == "__main__":
    main()

