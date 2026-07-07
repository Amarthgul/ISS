

from Util.Globals import RNG



class Coating:
    def __init__(self):

         self.reflectKill = 0



    def ReflectionModulate(self, reflectedRaybatch):

        return reflectedRaybatch








class NaiveCoating(Coating):

    def __init__(self):
         super().__init__()

         self.reflectKill = 0.6



    def ReflectionModulate(self, reflectedRaybatch):

        # Randomly drop some of the reflected rays based on reflectKill
        if reflectedRaybatch is None or reflectedRaybatch.value is None:
            return reflectedRaybatch

        rayCount = reflectedRaybatch.value.shape[0]
        if rayCount == 0:
            return reflectedRaybatch

        dropRate = min(max(float(self.reflectKill), 0.0), 1.0)
        keepMask = RNG.rand(rayCount) >= dropRate
        reflectedRaybatch.Mask(keepMask)

        return reflectedRaybatch


