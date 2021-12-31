# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from timeit import default_timer as timer
import glob
import numpy as np

# Own classes
from output import Output


def pipeline(size=10,
             circular=False,
             clocklike=False,
             predefinedSimulationMatrix=None,
             measurePerformance=False,
             measureDivergence=False,
             passLeafes=None,
             output=None,
             first_candidate_only=False,
             print_info=False):

    if measurePerformance:
        startTime = timer()

    simulationMatrix = None

    if (predefinedSimulationMatrix is None):
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        simulationMatrix = scenario.D
        # print(scenario.D)

    else:
        # use supplied matrix
        simulationMatrix = predefinedSimulationMatrix
        
    recognizeWrapper(
            simulationMatrix,
            first_candidate_only=first_candidate_only,
            print_info=print_info,
            measurePerformance=measurePerformance,
            measureDivergence=measureDivergence,
            passLeafes=passLeafes,
            output=output)        

    if measurePerformance:
        # measure time
        endTime = timer()
        output.measuredRuntime = endTime - startTime


    # print single outputs if needed
    # output.print()


def recognizeWrapper(D,
                     first_candidate_only=False,
                     print_info=False,
                     measurePerformance=False,
                     measureDivergence=False,
                     passLeafes=None,
                     output=None):
    # TODO: Somehow use the recognition_tree of recognize() to compare its first four leafes with the passed ones, calculate divergence and check for rmap
    recognition_tree = recognize(D, first_candidate_only, print_info)
    
    # recognition_tree.visualize()

    # You can imagine the root of this tree as all vertices given by the simulated matrix. Then it tries to reconstruct r-steps
    # by deleting a node and recalculates the distances and checks the laws of r-matrices. This is the way how it constructs its childs.



    # leafes - we will traverse the tree and check for its treenodes with only four vertices. We will compare them to the passed ones.  
    # maybe if we compare the distance matrices?? The problem is: In the simulation vertices are created with increasing numbers. 
    # therefore the first four leafes are always 0,1,2,3. So my idea is to use the distance matrix of the original four leafes and check for that.
    
    leafes_match = False
    if passLeafes is not None:
        for current_node in recognition_tree.preorder():
            if current_node.n == 4 and current_node.valid_ways == 1:
                # Compare matrix current_node.D and matrix passLeafes right here (write a function for this, both are 4x4 and symmetric). 
                # If we find one match return true on leafes_match
                # print(current_node.D)
                # if np.array_equal(passLeafes, current_node.D):
                #     leafes_match = True
                    # Problem arises: Even for the same path, the resulting matrices are not equal! This means we really have to check the list of nodes.
        
                # other way: check current list of vertices against list of  which was passed in passLeafes
                if sorted(current_node.V) == sorted(passLeafes):
                    leafes_match = True
                    print("Leafes matched!")


    # divergence TODO Here we will maybe need to compare the history of the original one and the r-steps of the treenodes somehow.
    # Tactic: Lets collect all r-steps from the reconstructed tree in a set. Then we compare this set to the original steps. So we will see
    # if there are steps which could not be reconstructed.
    current_divergence = 0
    if measureDivergence: 
        pass
    
    # RMap - we can check this with the valid_ways attribute of the root node
    was_classified_as_R_Map = False

    if recognition_tree.root.valid_ways > 0 :
        was_classified_as_R_Map = True
    
    # print("Valid ways of the root-Node: {}".format(recognition_tree.root.valid_ways))

    # Set corresponding values in the current output-Object
    output.divergence = current_divergence
    output.classifiedMatchingFourLeaves = leafes_match
    output.classifiedAsRMap = was_classified_as_R_Map

    return


def testOutputClass():
    a = Output(classifiedAsRMap=True,
               classifiedMatchingFourLeaves=True,
               divergence=10,
               measuredRuntime=10,
               plotMatrix=False,
               plotSteps=False,
               plotBoxPlots=False)
    return a


def wp2benchmark():
    '''
    for every matrix which was generated: Load it, and use the
    pipeline on it. Generate a new Output-Object for every of them
    and sum up Runtimes etc.
    '''
    # TODO: Ladebalken yikes

    # init values
    overallRuntime = 0.0
    numberOfRMaps = 0.0
    numberOfMatchingFourLeafs = 0.0
    sumOfDivergence = 0.0

    # load the files
    # path = '../../test-matrices/hists/*.txt'
    path = '../../test-matrices/subtest/*.txt'
    # path = '../../test-matrices/singletest/*.txt'
    filePaths = glob.glob(path)
    # print(len(filePaths))

    # get overall number of used scenarios
    numberOfScenarios = len(filePaths)

    # loop it baby, loop it!
    for currentPath in filePaths:
        # extract clockwise and circular info from filename
        fileName = currentPath[(len(currentPath)-12):len(currentPath)]
        # print(fileName)

        circular = False
        clocklike = False

        if (fileName.find("i") != -1):
            circular = True

        if (fileName.find("o") != -1):
            clocklike = True

        # Debug
        # if (circular):
        #    print("Is circular.")
        # if (clocklike):
        #    print("Is clocklike.")

        # load the corresponding matrix with a new Output Object
        scenario = load(filename=currentPath)
        # fourLeafScenario = load(filename=currentPath, stop_after=4)
        print(str(currentPath))
        # did it work? - obviously yes!
        # print(scenario.N)

        currentOutput = Output()

        # use the pipeline on it
        pipeline(size=scenario.N,
                 clocklike=clocklike,
                 circular=circular,
                 predefinedSimulationMatrix=scenario.D,
                 measurePerformance=True,
                 measureDivergence=True,
                 passLeafes=[0,1,2,3],
                 first_candidate_only=True,
                 output=currentOutput)
        # Note: PassedLeafs is none but divergence is true,
        # does this result in difficulties?

        # use the modified values of the Output Object to
        # modify overall values of benchmark
        if (currentOutput.classifiedAsRMap):
            numberOfRMaps += 1
        if (currentOutput.classifiedMatchingFourLeaves):
            numberOfMatchingFourLeafs += 1
        sumOfDivergence += currentOutput.divergence
        overallRuntime += currentOutput.measuredRuntime

        # print("Aimed for this matrix:")
        # print(fourLeafScenario.D)
        

    # return the benchmark results in a nice format
    print("\n\n------------WP2Benchmark------------------")
    print("Number of simulated matrices: {}".format(numberOfScenarios))
    print("Overall runtime measured: {} seconds needed.".format(overallRuntime))
    print("Proportion of classified R-Maps is: {}"
          .format(numberOfRMaps/numberOfScenarios))
    print("Proporion of 4-leaf-maps: {}"
          .format(numberOfMatchingFourLeafs/numberOfScenarios))
    print("Average divergence is: {}"
          .format(sumOfDivergence / numberOfScenarios))
    print(" End of the Benchmark ")


def testFileLoad():
    path = '../../test-matrices/hists/*.txt'
    files = glob.glob(path)
    for file in files:
        print(file)
