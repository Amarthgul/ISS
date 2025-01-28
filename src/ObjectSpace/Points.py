



from Util.Backend import backend as bd
from Util.Misc import RGBToWavelengthSameD, Lumi
from Util.Globals import ONE, INIT_PHASE_DIFF
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

        # Accquire the luminisity of the sources
        lumi = Lumi(source.Color())
        # Create a mask that prune the rays based on lumi
        random_mask = bd.random.random((lenSource, lenTarget)) < lumi[:, bd.newaxis]


        sourceExpanded = sourcePos[:, bd.newaxis, :]  # Shape (n, 1, 3)
        targetsExpanded = targets[bd.newaxis, :, :]  # Shape (1, m, 3)

        # Compute the direction of acrossing the source and target 
        dirCross = targetsExpanded - sourceExpanded # Shape (n, m, 3)
        
        # Expand and append the position into pos/dir pairs 
        sourcePos = sourcePos[:, bd.newaxis, :]
        sourcePos = bd.tile(sourcePos, (1, dirCross.shape[1], 1))
        appended = bd.concatenate([sourcePos, dirCross], axis=2)[random_mask]
        # After applying the mask, appended is of shape (m*n', 6)

        # Convert source color to wavelength 
        wavelengths, radiants = RGBToWavelengthSameD(source.Color())
        # Expand the wavelength to match the pos/dir 
        wavelengths = wavelengths[:, bd.newaxis, :]
        wavelengths = bd.tile(wavelengths, (1, dirCross.shape[1], 1))[random_mask]
        # After applying the mask, wavelengths is of shape (m*n', 3)

        # Spilt the wavelengths, copy and concatenate them to 
        wavelengths = bd.split(wavelengths, indices_or_sections=3, axis=1)
        appended = [bd.concatenate([appended, b], axis=1) for b in wavelengths]
        appended = bd.concatenate(appended, axis=0) # This yields a (3*m*n', 7) array 
        
        temp = bd.ones(3)
        temp[0] = ONE    # Sagittal radiant
        temp[1] = ONE    # Tangential radiant
        temp[2] = INIT_PHASE_DIFF   # Phase difference 

        return RayBatch(
            bd.concatenate([appended, bd.tile(temp, (appended.shape[0], 1))], axis=1)
        )


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

    print(RB.value)



if __name__ == "__main__":
    main() 

