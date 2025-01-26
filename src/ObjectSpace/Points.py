



from Util.Backend import backend as bd
from Raytracing.RayBatch import RayBatch




class PointsSource:
    """
    Point sources are organized in the form of:
    [[x, y, z, R, G, B], 
       [...], [...], ...]
    Alternatively, it may also be using field angle representation:
    [[θ_x, θ_y, D, R, G, B], 
       [...], [...], ...]] 
    Where D is the distance from the front vertex of the lens, which in this case is the polar origin.
    """

    def __init__(self, data):
        self.value = data

        """Whether the data is Cartesian XYZ coordinates or field angles"""    
        self.isCartesian = False

        """Whether the angle is in radian"""
        self.angleInRad = False



    def SetPoints(self, points):
        self.value = points


    def AddPoint(self, point):
        bd.vstack(self.value, point)


    def EmitTowards(self, targets):
        """
        Emit rays from the point sources towards the target points. 

        :param targets: collection of points as emission targets. 

        :return: raybatch object of rays from the point sources to the target, with corresponding  
        """

        pass 


