
"""
This module records the path of the rays so that some debugging and inpection features are eaiser to perform. 

It is not recommended to use this module in the ray tracing process of production imagings as it may signicantly increase the computational cost and memory usage.

"""

from Util.Backend import backend as bd
from Util.PlotTest import DrawLines, DrawNormal



class RayPath():
    def __init__(self):
        self.value = []
        self.reflected = []
        self.vignetted = []

        self._direction = []

    def Append(self, raybatch, reflected = None, vignetted = None):
        """
        Append a raybatch to the path. 

        :param raybatch: The raybatch to be recorded.
        :param reflected: bool array of reflected rays.
        :param vignetted: bool array of vignetted rays.
        """

        if (self.value is None):
            self.value = raybatch.Position()
            self._direction = raybatch.Direction()
        else:
            self.value.append(raybatch.Position())
            self._direction.append(raybatch.Direction())

        if (self.reflected is None):
            self.reflected = bd.ones_like(self.value).astype(bd.bool_)
        else:
            self.reflected.append(reflected)

        if(self.vignetted is None):
            self.vignetted = bd.ones_like(self.value).astype(bd.bool_)
        else:
            self.vignetted.append(vignetted)


    def PlotPath(self, expendEnd = 0.0):
        """
        Draw the path of the recorded rays. 
        """
        
        if(len(self.value) <= 1):
            raise ValueError("The path is too short to plot.")

        for i in range(len(self.value) - 1):
            # Bool filter out the rays that are vignetted or reflected 
            # so that both set of points have the same size 
            DrawLines(
                self.value[i][~self.vignetted[i+1]][~self.reflected[i+1]],    
                self.value[i+1], 
                lineColor = 'red', 
                lineWidth = 0.5
            )

        # Draw the last point
        if(expendEnd > 0):
            i = len(self.value)-1
            DrawNormal(
                self.value[i][~self.reflected[i]], 
                self._direction[i][~self.reflected[i]], 
                lineColor = 'red',
                lineLength = expendEnd
                )



















def main():
    pass 


if __name__ == "__main__":
    main()

