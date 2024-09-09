


class Lens:
    def __inti__(self):
        self.elements = []
        
    def UpdateLens(self):
        """
        Iterate throught the elements and update them
        """
        
        currentT = 0

        for e in self.elements:
            e.SetCumulative(currentT)
            currentT += e.thickness


