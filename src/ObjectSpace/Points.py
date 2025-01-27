



from Util.Backend import backend as bd
from Util.Misc import RGBToWavelengthSameD, Lumi
from Raytracing.RayBatch import RayBatch




class PointsSource:
    """
    Point sources are organized in the form of:
    [[x, y, z, R, G, B], 
       [...], [...], ...]
    RGB must be float number in the range of [0, 1]. 
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


        self.sampleRecord = None 



    def SetPoints(self, points):
        self.value = points


    def AddPoint(self, point):
        bd.vstack(self.value, point)


    def Position(self):
        return self.value[:, :3]
    

    def Color(self):
        return self.value[:, 3:]


    def EmitTowards(self, source, targets):
        """
        Emit rays from the point sources towards the target points. 

        
        :param targets: collection of points as emission targets. 

        :return: raybatch object of rays from the point sources to the target, with corresponding wavelengths. 
        """
        sourcePos = source.Position()

        lenSource = sourcePos.shape[0]
        lenTarget = targets.shape[0]

        sourceExpanded = sourcePos[:, bd.newaxis, :]  # Shape (n, 1, 3)
        targetsExpanded = targets[bd.newaxis, :, :]  # Shape (1, m, 3)

        # Compute the cross product pairwise and reshape it 
        cross_vectors = bd.cross(sourceExpanded, targetsExpanded)
        #cross_vectors = cross_vectors.reshape(-1, 3)  
        
        wavelengths, radiants = RGBToWavelengthSameD(source.Color())
        wlExpanded = wavelengths[bd.newaxis, :, :]
        wlBroadcasted = bd.tile(wlExpanded, (cross_vectors.shape[0], 1, 1))


        # Accquire the luminisity of the sources
        lumi = Lumi(self.Color())
        random_mask = bd.random.random((lenSource, lenTarget)) < lumi[:, bd.newaxis]
        cross_vectors = cross_vectors[random_mask]

        



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
        [-2, 3, 25], 
        [1, -2, 25]
    ])

    RB = t.EmitTowards(t, targets)



if __name__ == "__main__":
    main() 

