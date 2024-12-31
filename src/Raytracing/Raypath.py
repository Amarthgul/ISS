
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
        """

        if self.value is None:
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

        if(len(self.value) <= 1):
            raise ValueError("The path is too short to plot.")

        for i in range(len(self.value) - 1):
            DrawLines(
                self.value[i][~self.vignetted[i+1]],     # Previous surface 
                self.value[i+1], # Current surface
                lineColor = 'red', 
                lineWidth = 0.5
            )



















def main():
    pass 


if __name__ == "__main__":
    main()

