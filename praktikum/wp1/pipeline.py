# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from timeit import default_timer as timer
import glob
import os

# Own classes
from output import Output


def pipeline(size=10, circular=False, clocklike=False, predefinedSimulationMatrix=None, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None, first_candidate_only = False, print_info=False):

    if measurePerformance:
        startTime = timer()  
        #print(startTime)  

    if (predefinedSimulationMatrix is None):
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        print(scenario.D)
        recognition_tree = recognizeWrapper(scenario.D, first_candidate_only=first_candidate_only, print_info=print_info, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)

        if measurePerformance:
            # measure time
            endTime = timer()
            #print(endTime)
            output.measuredRuntime = endTime - startTime
    else:
        # use supplied matrix
        recognition_tree = recognizeWrapper(predefinedSimulationMatrix, first_candidate_only=first_candidate_only, print_info=print_info, measurePerformance=measurePerformance, measureDivergence=measureDivergence, passLeafes=passLeafes, output=output)

        if measurePerformance:
            # measure time
            endTime = timer()
            #print(endTime)
            output.measuredRuntime = endTime - startTime
    # print single outputs if needed
    # output.print()

        


def recognizeWrapper(D, first_candidate_only=False, print_info=False, measurePerformance=False, measureDivergence=False, passLeafes=None, output=None):
    return recognize(D, first_candidate_only, print_info)


def testOutputClass():
    a = Output(classifiedAsRMap = True, classifiedMatchingFourLeaves = True, divergence = 10, measuredRuntime = 10, plotMatrix = False, plotSteps = False, plotBoxPlots = False)
    return a


def wp2benchmark(): 
    # TODO: Ladebalken yikes


    # for every matrix which was generated: Load it, and use the pipeline on it.
    # generate a new Output-Object for every of them and sum up Runtimes etc.


    # init values
    overallRuntime = 0.0 
    numberOfRMaps = 0.0
    numberOfMatchingFourLeafs = 0.0
    sumOfDivergence = 0.0

    # load the files
    #path = '../../test-matrices/hists/*.txt'
    path = '../../test-matrices/subtest/*.txt'
    filePaths = glob.glob(path)
    #print(len(filePaths))

    # get overall number of used scenarios
    numberOfScenarios = len(filePaths)

    # loop it baby, loop it!
    for currentPath in filePaths:
        # extract clockwise and circular info from filename
        fileName = currentPath[(len(currentPath)-12):len(currentPath)]
        #print(fileName)

        circular = False
        clocklike = False

        if (fileName.find("i") != -1) :
            circular = True

        if (fileName.find("o") != -1) : 
            clocklike = True
        
        
        # Debug
        #if (circular):
        #    print("Is circular.")
        #if (clocklike):
        #    print("Is clocklike.")
        
        
        # load the corresponding matrix with a new Output Object
        scenario = load(filename=currentPath)
        print(str(currentPath))
        # did it work? - obviously yes! 
        # print(scenario.N)

        currentOutput = Output()
        
        # use the pipeline on it 
        
        pipeline(size = scenario.N, clocklike = clocklike, circular = circular, predefinedSimulationMatrix = scenario.D,
                    measurePerformance = True, measureDivergence = True, first_candidate_only = True,
                    output = currentOutput)
                # Note: PassedLeafs is none but divergence is true, does this result in difficulties?

        # use the modified values of the Output Object to modify overall values of benchmark
        if (currentOutput.classifiedAsRMap):
            numberOfRMaps += 1
        if (currentOutput.classifiedMatchingFourLeaves):
            numberOfMatchingFourLeafs += 1
        sumOfDivergence += currentOutput.divergence
        overallRuntime += currentOutput.measuredRuntime

    # return the benchmark results in a nice format
    print("------------WP2Benchmark------------------")
    print("Overall runtime measured: " + str(overallRuntime))
    print("Proportion of classified R-Maps is: " + str(numberOfRMaps/numberOfScenarios))
    print("Proporion of 4-leaf-maps: " + str(numberOfMatchingFourLeafs/numberOfScenarios))
    print("Average divergence is: " + str(sumOfDivergence / numberOfScenarios))
    print(" End of the Benchmark ")


def testFileLoad():
    path = '../../test-matrices/hists/*.txt'
    files = glob.glob(path)
    for file in files:
        print(file)