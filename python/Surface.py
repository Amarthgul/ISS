




class Surface:
    def __init__(self, r, t, d, m):
        self.radius = r
        self.thickness = t
        self.clearSemiDiameter = d 
        self.material = m
        
        self.chamfer = None 

        self.cumulativeThickness = None 
        self.frontVertex = None 
        self.origin = None 

        # Add asph 

        def SetCumulative(self, cd):
            self.cumulativeThickness = cd 
        
