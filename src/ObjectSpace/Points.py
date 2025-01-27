



from Util.Backend import backend as bd
from Util.Misc import RGBToWavelengthArray
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


    def Position(self):
        return self.value[:, :3]
    

    def Color(self):
        return self.value[:, 3:]


    def EmitTowards(self, targets, monochannel=None):
        """
        Emit rays from the point sources towards the target points. 

        :param targets: collection of points as emission targets. 
        :param monochannel: 

        :return: raybatch object of rays from the point sources to the target, with corresponding  
        """
        selfPos = self.Position()
        selfPosExpanded = selfPos[:, bd.newaxis, :]  # Shape (n, 1, 3)
        targetsExpanded = targets[bd.newaxis, :, :]  # Shape (1, m, 3)

        # Compute the cross product pairwise and reshape it 
        cross_vectors = bd.cross(selfPosExpanded, targetsExpanded).reshape(-1, 3)  
        
        print(self.Color())
        # TODO: create a new method that returns wavelength array with same shape 
        wavelengths, radiants = RGBToWavelengthArray(self.Color())
        

        print("  ")


def main():
    
    t = PointsSource(bd.array([
        [0,  0, -10000, 1, 1, 1], 
        [0, 10, -10000, 1, 0.6, 0.6],
        [0, 25, -10000, 1, 0.6, 0.3],
    ]))

    targets = bd.array([
        [1, 2, 25], 
        [2, 4,25],
        [-2, 3, 25]
    ])

    RB = t.EmitTowards(targets)



if __name__ == "__main__":
    main() 

