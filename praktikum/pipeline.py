# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from timeit import default_timer as timer
import glob
import numpy as np
import itertools

# Own classes
from output import Output

# TODO: Last point of WP1 is left.

def pipeline(size=10,
             circular=False,
             clocklike=False,
             predefinedSimulationMatrix=None,
             measurePerformance=False,
             measureDivergence=False,
             firstLeaves=None,
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
            measureDivergence=measureDivergence,
            history=history,
            firstLeaves=firstLeaves,
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
                     measureDivergence=False,
                     history=None,
                     firstLeaves=None,
                     skipLeaves=False,
                     forbiddenLeaves=None,
                     output=None):
    
    # Shall recognize skip forbidden leafes?
    if skipLeaves:
        recognition_tree = recognize(D, first_candidate_only, print_info, forbiddenLeaves)
    else: 
        recognition_tree = recognize(D, first_candidate_only, print_info)
    
    recognition_tree.visualize()
    
    # Check: Match recosntructed leafes and orginal leafes?
    leafes_match = False
    if firstLeaves is not None:
        for current_node in recognition_tree.postorder():
            if current_node.n == 4 and current_node.valid_ways == 1:
                
                # Check current list of vertices against passLeafes-list
                if sorted(current_node.V) == sorted(firstLeaves):
                    leafes_match = True
                    print("Leafes matched!")


    # Check: Do the R-Steps from the reconstructed tree diverge from the original R-Steps?
    current_divergence = 0.0
    if measureDivergence: 
        reconstructed_r_steps = set()
        
        for current_node in recognition_tree.preorder():    

            # add all r_steps where the result was an r-map at the end
            if current_node.valid_ways > 0 and current_node.R_step is not None:
                
                # Construct a new R-step which is comparable to those in the history
                temp = (current_node.R_step[0], current_node.R_step[1], current_node.R_step[2], float("{:.6f}".format(current_node.R_step[3])))
                # print("modified r-step:")
                # print(newTople)
                reconstructed_r_steps.add(temp)

        # print("R-Steps: " + str(reconstructed_r_steps))
        # Now we need to check the reconstructed r-steps against the original ones.
        # Extract r-steps from history

        history_r_steps = set()
        offset_counter = 0
        for entry in history:
            # print("Entry: " + str(entry))

            # Skip the first 3 entries since they aren't in the reconstructed set.
            if offset_counter <= 2:
               offset_counter += 1 
               # print("Skipped")
               continue

            # We have to modifiy the values and sort x,y in the same order (ascending) as the reconstructed r_steps are. 
            # The last entry of alpha does not match on the last few digits sometimes. So I restricted it to 6 digits.
            # print("Real entry")
            # print(float("{:.6f}".format(entry[3])))
            if entry[0] > entry[1]:
                newTuple=(entry[1], entry[0], entry[2], float("{:.6f}".format(1-entry[3])))
            else: 
                newTuple=(entry[0], entry[1], entry[2], float("{:.6f}".format(entry[3])))
            
            # print("New Tuple is: " + str(newTuple))
            
            history_r_steps.add(newTuple)
            

        # print("History-R-Steps: " + str(history_r_steps))
        # Now compare them. We use intersection to find elements that were contained in both.
        result = history_r_steps.intersection(reconstructed_r_steps)
        # print("Result intersection: " + str(result))
 
        # return the result as one minus the proportion of successful reconstructed steps from all original steps. Care for cases with n=4.
        if len(history_r_steps) != 0 :
            current_divergence =  1 - (len(result) / len(history_r_steps))
            # print("Current divergence: " + str(current_divergence))


    # Check: Was the simulated Matrix a R-Map?
    was_classified_as_R_Map = False
    if recognition_tree.root.valid_ways > 0 :
        was_classified_as_R_Map = True
    # If not, Reconstruction failed and we should TODO: output "plot distance matrices, recognition steps and final box plots of scenarios"
    else: 
        pass
    
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


def benchmark(workPackage=2, firstLeaves=[0,1,2,3], skipLeaves=False, forbiddenLeaves=None):
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

    # Load the files
    # path = '../test-matrices/hists/*.txt'
    path = '../test-matrices/subtest/*.txt'
    # path = '../test-matrices/singletest/*.txt'
    filePaths = glob.glob(path)


    # Get overall number of used scenarios
    numberOfScenarios = len(filePaths)
    # print(len(filePaths))

    # For every file, use the pipeline ~ loop it baby, loop it!
    for currentPath in filePaths:
        print("Current File is: " + str(currentPath))

        # extract clockwise and circular info from filename
        fileName = currentPath[(len(currentPath)-12):len(currentPath)]
        # print(fileName)

        circular = False
        clocklike = False

        if (fileName.find("i") != -1):
            circular = True

        if (fileName.find("o") != -1):
            clocklike = True


        # Load the corresponding matrix with a new Output Object
        scenario = load(filename=currentPath)


        # Write here the Wrapper which shall guess the core leaves and tries to avoid them in the recognition. Run this until you find a valid R-Map.
        # scenario.N has the number of items which were generated. So we need all subsets of N items with 3 respectively 4 leaves.
        combinationsOfThreeLeafes = list(itertools.combinations([x for x in range(scenario.N)],3))
        combinationsOfFourLeafes = list(itertools.combinations([x for x in range(scenario.N)],4))

        # Rotate until you find a valid solution
        foundValidRMap = False
        # TODO: Use the different items of the combinations in forbiddenLeaves
        while not foundValidRMap:

            # Create our Object where the evaluation will be captured.  
            currentOutput = Output()
            # use the pipeline on it
            pipeline(size=scenario.N,
                     clocklike=clocklike,
                    circular=circular,
                    predefinedSimulationMatrix=scenario.D,
                    measurePerformance=True,
                    measureDivergence=True,
                    firstLeaves=firstLeaves,
                    first_candidate_only=True,
                    history=scenario.history,
                    forbiddenLeaves=forbiddenLeaves,
                    skipLeaves=skipLeaves,
                    output=currentOutput)
            if currentOutput.classifiedAsRMap:
                foundValidRMap = True
            else: 
                # Normally rotate through the forbiddenLeaves but this is not yet implemented, so: TODO
                foundValidRMap = True

        # Use the values of the current Output Object to modify overall values of benchmark
        if (currentOutput.classifiedAsRMap):
            numberOfRMaps += 1
        if (currentOutput.classifiedMatchingFourLeaves):
            numberOfMatchingFourLeafs += 1
        sumOfDivergence += currentOutput.divergence
        overallRuntime += currentOutput.measuredRuntime

    # Return the benchmark results in a nice format
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
    path = '../test-matrices/hists/*.txt'
    files = glob.glob(path)
    for file in files:
        print(file)

def wp2benchmark():
    benchmark(workPackage=2)

def wp3benchmark():

    # Determine which Leaves are forbidden to sort out in the recognition algorithm
    benchmark(workPackage=3, skipLeaves=True, forbiddenLeaves=[0,1,2,3])

    # TODO: Last point of WP3
    # TODO: Do we have to split the implementation for 3/4 vertices? Or will we first try it without 4 vertices and afterwards without 3?
  