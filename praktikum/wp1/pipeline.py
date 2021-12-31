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
             history=None,
             output=None,
             first_candidate_only=False,
             skipLeaves=False,
             forbiddenLeaves=None,
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
            history=history,
            passLeafes=passLeafes,
            skipLeaves=skipLeaves,
            forbiddenLeaves=forbiddenLeaves,
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
                     history=None,
                     passLeafes=None,
                     skipLeaves=False,
                     forbiddenLeaves=None,
                     output=None):
    # Somehow use the recognition_tree of recognize() to compare its first four leafes with the passed ones, calculate divergence and check for rmap
    
    if skipLeaves:
        # extract starting leaves from scenarios history
        # use recognize with those forbidden leaves
        recognition_tree = recognize(D, first_candidate_only, print_info, forbiddenLeaves)
    else: 
        recognition_tree = recognize(D, first_candidate_only, print_info)
    
    recognition_tree.visualize()

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


    # divergence - Here we will maybe need to compare the history of the original one and the r-steps of the treenodes somehow.
    # Tactic: Lets collect all r-steps from the reconstructed tree in a set. Then we compare this set to the original steps. So we will see
    # if there are steps which could not be reconstructed.
    current_divergence = 0.0

    if measureDivergence: 
        reconstructed_r_steps = set()
        for current_node in recognition_tree.preorder():    
            if current_node.valid_ways > 0 and current_node.R_step is not None:
                # add all r_steps where the result was an r-map at the end
                # but construct an new r-step which is comparable with them of the history
                temp = (current_node.R_step[0], current_node.R_step[1], current_node.R_step[2], float("{:.6f}".format(current_node.R_step[3])))
                # print("modified r-step:")
                # print(newTople)
                reconstructed_r_steps.add(temp)

        # print("R-Steps")
        # print(reconstructed_r_steps)
        # now we need to check the reconstructed r-steps against the original ones.
        # print("History")
        # extract r-steps from history
        history_r_steps = set()
        offset_counter = 0
        for entry in history:
            # TODO: we have to skip the first four entries since we stop check for r-steps on 4 vertices
            # print("Entry:")
            # print(entry)
            # print("resulting tuple as r-step:")
            # we have to modifiy the values and sort them in the same order as the reconstructed r_steps are. (ascending)
            # print(float("{:.6f}".format(entry[3])))
            if entry[0] > entry[1]:
                newTuple=(entry[1], entry[0], entry[2], float("{:.6f}".format(1-entry[3])))
            else: 
                newTuple=(entry[0], entry[1], entry[2], float("{:.6f}".format(entry[3])))
            # print(newTuple)
            # next problem: The last entry of alpha does not match on the last few digits sometimes. So I restricted it to 6 digits.
            # skip first three entries
            if offset_counter > 2:
                history_r_steps.add(newTuple)

            offset_counter += 1

        # print(history_r_steps)
        # now compare them. We use intersection to find elements that were contained in both.
        result = history_r_steps.intersection(reconstructed_r_steps)
        # print("Result intersection")
        # print(result)
        # return the result as one minus the proportion of successful reconstructed steps from all original steps. Care for cases with n=4.
        if len(history_r_steps) != 0 :
            current_divergence =  1 - (len(result) / len(history_r_steps))


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


def benchmark(workPackage=2, skipLeaves=False, forbiddenLeaves=None):
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

        # Create our Object where the evaluation will be captured.
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
                 history=scenario.history,
                 forbiddenLeaves=forbiddenLeaves,
                 skipLeaves=skipLeaves,
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
    print("\n\n------------WP{}Benchmark------------------".format(workPackage))
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

def wp2benchmark():
    benchmark(workPackage=2, skipLeaves=False)

def wp3benchmark():

    # Determine which Leaves are forbidden to sort out in the recognition algorithm
    benchmark(workPackage=3, skipLeaves=True, forbiddenLeaves=[0,1,2,3])

    # TODO: Last point of WP3
    # TODO: Do we have to split the implementation for 3/4 vertices? Or will we first try it without 4 vertices and afterwards without 3?
    # Maybe this needs to be nested inside the pipeline since thats the place where we know how many vertices 