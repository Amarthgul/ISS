

from PIL import Image
import time
import matplotlib.pyplot as plt

from Util.Backend import backend as bd
from Util.ColorWavelength import ImageConversion
from Util.PltPlot import DrawRaybatch, AddXYZ, SetUnifScale, DrawPoints
from ExampleLenses import Biotar50mmf14, Helios58mmf2, CanonFD50mmf18
from Imagers.Standard import StdImager 
from ObjectSpace.Points import PointsSource
from ObjectSpace.Images import Image2D
from Raytracing.Emission import EmitField



def SpotTesting(objectDistance = 10000, focusDistance = 20000, saveIterationCount = 50):

    lens = Biotar50mmf14()

    source = PointsSource()
    source.isCartesian = False
    source.GenerateSpots(19, 12, dist=objectDistance)

    imager = StdImager(lens.BestFocusBFD(focusDistance), horiPx=9000) #32.4
    imager.SetLensLength(lens.totalAxialLength)
    image = imager.AccquireEmpty() 

    start = time.time()

    # plt.ion()  # Turn on interactive mode
    # fig, ax = plt.subplots()
    # im = ax.imshow(ImageConversion(image))
    iterationCount = 0

    while(True):
        mainRB = source.EmitSamplesToward(lens.entrancePupil.GetSamplePoints(20480), 5, addSecondary=9)

        lens.SetIncidentRaybatch(mainRB)

        mainRB, mainRP = lens.Propagate()

        mainRB, _tir, _vig = imager.IntersectRays(mainRB)
        # mainRP.Append(mainRB, _tir, _vig)

        image = imager.IntegralRays(mainRB, baseImg=image)

        
        # im.set_data(ImageConversion(image, amplifier=0.25))
        # plt.draw()
        # plt.pause(0.01)
        
        #print(source.sampleRecord)
        elpased = time.time() - start

        print("- Focusing ", focusDistance, " for obj at ", objectDistance,  
            "  \t\tAt ", str(iterationCount), "th iteration after ", str(elpased) )

        if(iterationCount > saveIterationCount):
            imgSave = Image.fromarray(ImageConversion(image), 'RGB')
            imgSave.save(r"resources/Results/SpotTestng/ObjDist"+str(objectDistance)+"_FocusDist"+str(focusDistance)+ "_RID"+str(elpased) + ".png")
            break

        iterationCount += 1



def main():

    objectDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]
    focusDistance = [
        350, 500, 800, 1200, 1500, 2000, 3000, 5000, 8000, 15000, 30000, 100000
    ]

    for obj in objectDistance:
        for focus in focusDistance:
            SpotTesting(obj, focus, 512)
    
    


if __name__ == "__main__":
    main()