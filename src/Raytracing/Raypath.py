
"""
This module records the path of the rays so that some debugging and inpection features are eaiser to perform. 

"""


from Util.Backend import backend as bd
from Util.PlotTest import DrawLines

class RayPath():
    def __init__(self):
        self.value = []
        self.reflected = []
        self.vignetted = []


    def Append(self, raybatch, reflected, vignetted):
        """
        Append a raybatch to the path. 

        :param raybatch: The raybatch to be recorded.
        :param reflected: bool array of reflected rays.
        :param vignetted: bool array of vignetted rays.
        """

        if (self.value is None):
            self.value = raybatch.Position()
        else:
            self.value.append(raybatch.Position())

        if (self.reflected is None):
            self.reflected = bd.zeros_like(self.value)
        else:
            self.reflected.append(reflected)

        if(self.vignetted is None):
            self.vignetted = bd.zeros_like(self.value)  
        else:
            self.vignetted.append(vignetted)


    def PlotPath(self):
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



















def main():
    pass 


if __name__ == "__main__":
    main()

