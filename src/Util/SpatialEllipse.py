




class SpatialEllipse():
    """
    This class is mostly for marking and calculating the clear boundary. 
    """
    # Although the name may be associated with the polarized raidance ellipse that pop up often in this project, this class is not for that thing. 
    # The polarized raidance ellipse is 2D for simplicity and because the polarization is local to the ray, but this spatial ellipse is 3D. 

    def __init__(self, C, u, v, a, b):
        self.center = C
        self.semiAxisDirU = u
        self.semiAxisDirV = v
        self.semiAxisMagA = a
        self.semiAxisMagB = b 

        self.isCircular = True

    def FacingDirection(self):
        return self.semiAxisDirU @ self.semiAxisDirV

