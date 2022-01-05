# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from typing import Final
from timeit import default_timer as timer
from logging import info
import itertools
import random
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Own classes
from output import Output

# Constants
WORK_PACKAGE_2: Final = 2
WORK_PACKAGE_3: Final = 3
WORK_PACKAGE_3_4: Final = 34

# TODO: Last point of WP1 is left.
# TODO: Parallelisierung
# TODO: Don't TelL StaDlEr!!11
# TODO: Output blanks implementieren
# Halbes TODO: Generation in pipeline überarbeiten
# (size, clocklike und circular)


def pipeline(size: int = 10,
             circular: bool = False,
             clocklike: bool = False,
             predefinedSimulationMatrix=None,
             measurePerformance: bool = False,
             measureDivergence: bool = False,
             firstLeaves=None,
             history=None,
             first_candidate_only: bool = True,
             forbiddenLeaves=None,
             print_info: bool = False) -> Output:

    if measurePerformance:
        startTime = timer()

    simulationMatrix = None

    if (predefinedSimulationMatrix is None):
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        simulationMatrix = scenario.D
        info(scenario.D)

    else:
        # use supplied matrix
        simulationMatrix = predefinedSimulationMatrix

    output = recognizeWrapper(
            simulationMatrix,
            first_candidate_only=first_candidate_only,
            print_info=print_info,
            measureDivergence=measureDivergence,
            history=history,
            firstLeaves=firstLeaves,
            forbiddenLeaves=forbiddenLeaves)

    if measurePerformance:
        # measure time
        endTime = timer()
        output.measuredRuntime = endTime - startTime

    # print single outputs if debug is enabled
    info(output)

    return output


def recognizeWrapper(D,
                     first_candidate_only=True,
                     print_info=False,
                     measureDivergence=False,
                     history=None,
                     firstLeaves=None,
                     forbiddenLeaves=None) -> Output:

    # Shall recognize skip forbidden leafes?
    if forbiddenLeaves is not None:
        recognition_tree = recognize(D, first_candidate_only, print_info,
                                     forbiddenLeaves)
    else:
        recognition_tree = recognize(D, first_candidate_only, print_info)

    # recognition_tree.visualize()

    # Check: Match reconstructed leafes and orginal leafes?
    leafes_match = False
    if firstLeaves is not None:
        # build a set which contains all childs
        possible_node_set = []

        for current_node in recognition_tree.postorder():
            if current_node.n == 4 and current_node.valid_ways == 1:
                possible_node_set.append(current_node)

        # Choose random one of the nodes as stated in WP2
        choosen_node = random.choice(possible_node_set)
        info(f'Randomly choosen last node: {choosen_node.V}')
        # Check current list of chosen_node against passLeafes-list
        if set(firstLeaves).issubset(set(choosen_node.V)):
            leafes_match = True
            info('Leafes matched!')

    # Check: Do the R-Steps from the reconstructed tree
    # diverge from the original R-Steps?
    divergence_with_order = 0.0
    divergence_without_order = 0.0
    if measureDivergence:
        reconstructed_r_steps = []

        for current_node in recognition_tree.postorder():

            # add all r_steps where the result was an r-map at the end
            if current_node.valid_ways > 0 and current_node.R_step is not None:

                # special case if we are at the end and have one of the
                # possible candidates, we only want that one which was chosen
                # randomly before at the leaf matching.
                # So we will skip all others.
                if current_node.n == 4 and current_node != choosen_node:
                    info(f'Skipped this entry: {str(current_node.V)} although',
                         'it was valid but not chosen.')
                    continue

                # Construct a new R-step which is comparable
                # to those in the history
                temp = (current_node.R_step[0],
                        current_node.R_step[1],
                        current_node.R_step[2],
                        round(current_node.R_step[3], 6))
                reconstructed_r_steps.append(temp)

        new_list = sorted(reconstructed_r_steps, key=lambda item: item[2])
        info(f'R-Steps from reconstruction:\n{str(new_list)}')
        # Now we need to check the reconstructed r-steps against
        # the original ones. Extract r-steps from history
        history_r_steps = []
        offset_counter = 0
        for entry in history:
            # Skip the first 3 entries since they
            # aren't in the reconstructed set.
            if offset_counter <= 2:
                offset_counter += 1
                continue

            # We have to modifiy the values and sort x,y in the
            # same order (ascending) as the reconstructed r_steps are.
            # The last entry of alpha does not match on the last
            # few digits sometimes. So I restricted it to 6 digits.
            if entry[0] > entry[1]:
                newTuple = (entry[1], entry[0], entry[2], round(1-entry[3], 6))
            else:
                newTuple = (entry[0], entry[1], entry[2], round(entry[3], 6))

            history_r_steps.append(newTuple)

        info(f'R-Steps from history:\n{str(history_r_steps)}')
        # Now compare them. We use intersection to find elements
        # that were contained in both.

        # Matching entrys with order
        matchCounter = 0
        for index in range(len(history_r_steps)):
            if history_r_steps[index] == reconstructed_r_steps[index]:
                matchCounter += 1
            info(f'Compare: {history_r_steps[index]} and {reconstructed_r_steps[index]}')
        info(f'Match Counter is: {matchCounter}')
        # Matching entrys without order
        result = set(history_r_steps).intersection(set(reconstructed_r_steps))

        # return the result as one minus the proportion of successful
        # reconstructed steps from all original steps. Care for cases with n=4.
        if len(history_r_steps) != 0:
            divergence_without_order = 1 - (len(result) / len(history_r_steps))
            divergence_with_order = 1 - (matchCounter / len(history_r_steps))
            info(f'Current divergence without order: {divergence_without_order}')
            info(f'Current divergence with order: {divergence_with_order}')

    # Check: Was the simulated Matrix an R-Map?
    was_classified_as_R_Map = False
    if recognition_tree.root.valid_ways > 0:
        was_classified_as_R_Map = True
    # If not, Reconstruction failed and we should
    # TODO: output 'plot distance matrices, recognition steps
    #       and final box plots of scenarios'
    else:
        pass

    info(f'Valid ways of the root-Node: {recognition_tree.root.valid_ways}')

    # Set corresponding values in the current output-Object
    output = Output()
    output.divergenceWithoutOrder = divergence_without_order
    output.divergenceWithOrder = divergence_with_order
    output.classifiedMatchingFourLeaves = leafes_match
    output.classifiedAsRMap = was_classified_as_R_Map

    return output


def benchmark(test_set: Path,
              workPackage=2,
              firstLeaves=[0, 1, 2, 3],
              forbiddenLeaves=None):
    '''
    for every matrix which was generated: Load it, and use the
    pipeline on it. Generate a new Output-Object for every of them
    and sum up Runtimes etc.
    '''
    # init values
    overallRuntime = 0.0
    numberOfRMaps = 0.0
    numberOfMatchingFourLeafs = 0.0
    sumOfDivergenceWithOrder = 0.0
    sumOfDivergenceWithoutOrder = 0.0

    # Load the files
    filePaths = list(test_set.glob('*'))

    # Get overall number of used scenarios
    numberOfScenarios = len(filePaths)

    # Prevent logging from messing up our progress bar!
    with logging_redirect_tqdm():
        # For every file, use the pipeline ~ loop it baby, loop it!
        for currentPath in tqdm(filePaths):
            tqdm.write(f'Current File is: {currentPath}')

            # extract clockwise and circular info from filename
            fileName = currentPath.name

            circular = False
            clocklike = False

            if (fileName.find('i') != -1):
                circular = True

            if (fileName.find('o') != -1):
                clocklike = True

            # Load the corresponding matrix with a new Output Object
            scenario = load(filename=currentPath)

            # Write here the Wrapper which shall guess the core leaves and
            # tries to avoid them in the recognition. Run this until you find
            # a valid R-Map.
            # scenario.N has the number of items which were generated.
            # So we need all subsets of N items with 3 respectively 4 leaves.

            # ForbiddenLeaves is an int at the end of WP3.4 and a list at WP3.3
            if type(forbiddenLeaves) is list:
                combinationsOfLeafes = [forbiddenLeaves]
            elif type(forbiddenLeaves) is int:
                possibleLeaves = range(scenario.N)
                leaveCombinations = itertools.combinations(possibleLeaves,
                                                           forbiddenLeaves)
                # We reverse the list because the solution [0,1,2,3]
                # is always trivial and the first.
                # Other solutions are more interesting.
                combinationsOfLeafes = reversed(list(leaveCombinations))
            else:
                combinationsOfLeafes = [None]

            # Rotate until you find a valid solution
            for combination in combinationsOfLeafes:
                info(f'Checked combination of core leafes: {str(combination)}')
                if combination is not None:
                    # The first leafes must correspond to the ones which
                    # are forbidden and therefore can't be deleted
                    # by the recognition algorithm.
                    firstLeaves = combination
                # Create our Object where the evaluation will be captured.
                currentOutput = Output()
                # use the pipeline on it
                currentOutput = pipeline(size=scenario.N,
                                         clocklike=clocklike,
                                         circular=circular,
                                         predefinedSimulationMatrix=scenario.D,
                                         measurePerformance=True,
                                         measureDivergence=True,
                                         firstLeaves=firstLeaves,
                                         first_candidate_only=True,
                                         history=scenario.history,
                                         forbiddenLeaves=combination)

                # Use the values of the current Output Object
                # to modify overall values of benchmark
                if (currentOutput.classifiedAsRMap):
                    numberOfRMaps += 1
                if (currentOutput.classifiedMatchingFourLeaves):
                    numberOfMatchingFourLeafs += 1
                sumOfDivergenceWithOrder += currentOutput.divergenceWithOrder
                sumOfDivergenceWithoutOrder += currentOutput.divergenceWithoutOrder
                overallRuntime += currentOutput.measuredRuntime

                # WP3 Stichpunkt vier.
                if (workPackage == WORK_PACKAGE_3_4
                   and currentOutput.classifiedAsRMap):
                    break

    # Return the benchmark results in a nice format
    print(f'\n\n------------WP{workPackage}Benchmark------------------')
    print(f'Number of simulated matrices: {numberOfScenarios}')
    print(f'Overall runtime measured: {overallRuntime} seconds needed.')
    proportion = numberOfRMaps / numberOfScenarios
    print(f'Proportion of classified R-Maps is: {proportion}')
    proportion = numberOfMatchingFourLeafs/numberOfScenarios
    print(f'Proporion of 4-leaf-maps: {proportion}')
    divergenceWithOrder = sumOfDivergenceWithOrder / numberOfScenarios
    print(f'Average divergence with order is: {divergenceWithOrder :.2%}')
    divergenceWithoutOrder = sumOfDivergenceWithoutOrder / numberOfScenarios
    print(f'Average divergence without order is: {divergenceWithoutOrder :.2%}')
    print(' End of the Benchmark ')


def wp2benchmark(test_set: Path):
    benchmark(test_set, workPackage=WORK_PACKAGE_2)


def wp31benchmark(test_set: Path):
    benchmark(test_set, workPackage=WORK_PACKAGE_3, forbiddenLeaves=[0, 1, 2])


def wp32benchmark(test_set: Path):
    benchmark(test_set,
              workPackage=WORK_PACKAGE_3,
              forbiddenLeaves=[0, 1, 2, 3])


def wp341benchmark(test_set: Path):
    benchmark(test_set, workPackage=WORK_PACKAGE_3_4, forbiddenLeaves=3)


def wp342benchmark(test_set: Path):
    benchmark(test_set, workPackage=WORK_PACKAGE_3_4, forbiddenLeaves=4)
