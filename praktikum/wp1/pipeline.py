from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize
import os

#TODO tests cases

def pipeline(size=10, circular=False, clocklike=False, predefinedSimulationMatrix=None, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None):

    if (predefinedSimulationMatrix == None):
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        print(scenario.D)
        recognition_tree = recognizeWrapper(scenario.D, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)
    else:
        # use supplied matrix
        recognition_tree = recognizeWrapper(predefinedSimulationMatrix, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)


def recognizeWrapper(D, first_candidate_only=False, print_info=False, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None):
    return recognize(D, first_candidate_only, print_info)
