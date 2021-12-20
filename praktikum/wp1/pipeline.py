# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from timeit import default_timer as timer
import os

# Own classes
from output import Output



#TODO tests cases @Malte
#TODO Output Klasse mit Performance Parameter usw, recognize steps usw.

def pipeline(size=10, circular=False, clocklike=False, predefinedSimulationMatrix=None, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None):

    if measurePerformance:
        startTime = timer()    

    if (predefinedSimulationMatrix == None):
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        print(scenario.D)
        recognition_tree = recognizeWrapper(scenario.D, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)

        if measurePerformance:
            # measure time
            endTime = timer()
            output.measuredRuntime = startTime - endTime
    else:
        # use supplied matrix
        recognition_tree = recognizeWrapper(predefinedSimulationMatrix, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)

        # print output
        # output.print()


def recognizeWrapper(D, first_candidate_only=False, print_info=False, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None):
    return recognize(D, first_candidate_only, print_info)


def testOutputClass():
    a = Output(classifiedAsRMap = True, classifiedMatchingFourLeaves = True, divergence = 10, measuredRuntime = 10, plotMatrix = False, plotSteps = False, plotBoxPlots = False)
    return a