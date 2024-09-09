




class Surface:
    def __init__(self, r, t, d, m):
        self.radius = None
        self.thickness = None
        self.material = None
        self.clearSemiDiameter = None 
        self.chamfer = None 

        self.cumulativeThickness = None 

        # Add asph 

        def SetCumulative(self, cd):
            self.cumulativeThickness = cd 
        
