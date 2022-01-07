# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize

# Python packages
from typing import Final, Optional, Union
from timeit import default_timer as timer
from logging import info
import itertools
import random
from pathlib import Path
from multiprocessing import Pool
from functools import reduce

# Own classes
from output import Output

# Constants
WORK_PACKAGE_2: Final = 2
WORK_PACKAGE_3: Final = 3
WORK_PACKAGE_3_4: Final = 34

History = list[tuple[int, int, int, float, float]]

# TODO: Last point of WP1 is left.
# TODO: Parallelisierung
# TODO: Don't TelL StaDlEr!!11
# TODO: Output blanks implementieren
# Halbes TODO: Generation in pipeline überarbeiten
# (size, clocklike und circular)


class BenchmarkStatistics:
    overallRuntime: float
    numberOfRMaps: int
    numberOfMatchingFourLeafs: int
    sumOfDivergenceWithOrder: float
    sumOfDivergenceWithoutOrder: float

    def __init__(self) -> None:
        self.overallRuntime = 0.0
        self.numberOfRMaps = 0
        self.numberOfMatchingFourLeafs = 0
        self.sumOfDivergenceWithOrder = 0.0
        self.sumOfDivergenceWithoutOrder = 0.0

    def add(left: 'BenchmarkStatistics',
            right: 'BenchmarkStatistics') -> 'BenchmarkStatistics':
        sum = BenchmarkStatistics()
        sum.overallRuntime = left.overallRuntime + right.overallRuntime
        sum.numberOfRMaps = left.numberOfRMaps + right.numberOfRMaps
        sum.numberOfMatchingFourLeafs = (left.numberOfMatchingFourLeafs +
                                         right.numberOfMatchingFourLeafs)
        sum.sumOfDivergenceWithOrder = (left.sumOfDivergenceWithOrder +
                                        right.sumOfDivergenceWithOrder)
        sum.sumOfDivergenceWithoutOrder = (
            left.sumOfDivergenceWithoutOrder +
            right.sumOfDivergenceWithoutOrder)
        return sum

    def pretty_print(self, total_count: int, work_package: int) -> None:
        # Return the benchmark results in a nice format
        correctly_classified = self.numberOfRMaps / total_count
        prop_4_leaf = self.numberOfMatchingFourLeafs / total_count
        divergence_ordered = self.sumOfDivergenceWithOrder / total_count
        divergence_unordered = self.sumOfDivergenceWithoutOrder / total_count

        print(f'''
  +--------------= Benchmark =---------------+
  |                  Workpackage: {work_package :>9}  |
  | Number of simulated matrices: {total_count :>9}  |
  |     Overall runtime measured: {self.overallRuntime :>9.2f}s |
  |  Correctly classified R-Maps: {correctly_classified :>10.2%} |
  |    Proportion of 4-leaf-maps: {prop_4_leaf :>10.2%} |
  |  Avg. divergence   (ordered): {divergence_ordered :>10.2%} |
  |  Avg. divergence (unordered): {divergence_unordered :>10.2%} |
  +------------------------------------------+
        ''')


class Benchmark:
    '''This is just a benchmark function with some state attached.
       Used to allow the easy usage of multiprocessing.Pool'''
    work_package: int
    forbidden_leaves: Union[list[int], int, None]

    def __init__(self,
                 work_package: int,
                 forbidden_leaves: Union[list[int], int, None]) -> None:
        self.work_package = work_package
        self.forbidden_leaves = forbidden_leaves

    def __call__(self, path: Path) -> BenchmarkStatistics:
        print(f'Working on: {path}')
        # extract clockwise and circular info from filename
        circular = path.name.find('i') != -1
        clocklike = path.name.find('o') != -1
        # Load the corresponding matrix with a new Output Object
        scenario = load(filename=path)
        # Write here the Wrapper which shall guess the core leaves and
        # tries to avoid them in the recognition. Run this until you find
        # a valid R-Map.
        # scenario.N has the number of items which were generated.
        # So we need all subsets of N items with 3 respectively 4 leaves.
        # ForbiddenLeaves is an int at the end of WP3.4 and a list at WP3.3
        combinationsOfLeafes = expand_leaves(self.forbidden_leaves, scenario.N)
        stats = BenchmarkStatistics()
        # Rotate until you find a valid solution
        for combination in combinationsOfLeafes:
            info(f'Checked combination of core leaves: {combination}')
            if combination is not None:
                # The first leaves must correspond to the ones which
                # are forbidden and therefore can't be deleted
                # by the recognition algorithm.
                first_leaves = combination
            # use the pipeline to create our output object
            output = pipeline(size=scenario.N,
                              clocklike=clocklike,
                              circular=circular,
                              predefinedSimulationMatrix=scenario.D,
                              measurePerformance=True,
                              measureDivergence=True,
                              first_leaves=first_leaves,
                              first_candidate_only=True,
                              history=scenario.history,
                              forbidden_leaves=combination)
            # Use the values of the current Output Object
            # to modify overall values of benchmark
            if (output.classifiedAsRMap):
                stats.numberOfRMaps += 1
            if (output.classifiedMatchingFourLeaves):
                stats.numberOfMatchingFourLeafs += 1
            stats.sumOfDivergenceWithOrder += output.divergenceWithOrder
            stats.sumOfDivergenceWithoutOrder += output.divergenceWithoutOrder
            stats.overallRuntime += output.measuredRuntime
            # WP3 Stichpunkt vier.
            # TODO: What was this about again?
            if (self.work_package == WORK_PACKAGE_3_4
               and output.classifiedAsRMap):
                break
        return stats


def pipeline(history: History,
             size: int = 10,
             circular: bool = False,
             clocklike: bool = False,
             predefinedSimulationMatrix: None = None,
             measurePerformance: bool = False,
             measureDivergence: bool = False,
             first_leaves: Optional[list[int]] = None,
             first_candidate_only: bool = True,
             forbidden_leaves: Optional[list[int]] = None,
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
            first_leaves=first_leaves,
            forbidden_leaves=forbidden_leaves)

    if measurePerformance:
        # measure time
        endTime = timer()
        output.measuredRuntime = endTime - startTime

    # print single outputs if debug is enabled
    info(output)

    return output


def recognizeWrapper(D: list[int],
                     history: History,
                     first_candidate_only: bool = True,
                     print_info: bool = False,
                     measureDivergence: bool = False,
                     first_leaves: Optional[list[int]] = None,
                     forbidden_leaves: Optional[list[int]] = None) -> Output:

    # Shall recognize skip forbidden leaves?
    if forbidden_leaves is not None:
        recognition_tree = recognize(D, first_candidate_only, print_info,
                                     forbidden_leaves)
    else:
        recognition_tree = recognize(D, first_candidate_only, print_info)

    # recognition_tree.visualize()

    # Check: Match reconstructed leaves and orginal leaves?
    leaves_match = False
    if first_leaves is not None:
        # build a set which contains all childs
        possible_node_set = []

        for current_node in recognition_tree.postorder():
            if current_node.n == 4 and current_node.valid_ways == 1:
                possible_node_set.append(current_node)

        # Choose random one of the nodes as stated in WP2
        choosen_node = random.choice(possible_node_set)
        info(f'Randomly choosen last node: {choosen_node.V}')
        # Check current list of chosen_node against passLeafes-list
        if set(first_leaves).issubset(set(choosen_node.V)):
            leaves_match = True
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
    output.classifiedMatchingFourLeaves = leaves_match
    output.classifiedAsRMap = was_classified_as_R_Map

    return output


def expand_leaves(leaves: Union[list[int], int, None],
                  count: int) -> list[list[int]]:
    ''' Convert given leaves into a list of list of leaves
        This is mainly used for the expansion of forbidden_leaves
    '''
    if type(leaves) is list:
        return [leaves]
    elif type(leaves) is int:
        possible_leafes = range(count)
        leav_combinations = itertools.combinations(possible_leafes, leaves)
        leav_lists = [list(tupl) for tupl in leav_combinations]
        # We reverse the list because the solution [0,1,2,3]
        # is always trivial and the first.
        # Other solutions are more interesting.
        return list(reversed(leav_lists))
    else:
        return []


def benchmark_all(test_set: Path,
                  work_package: int = 2,
                  first_leaves: list[int] = [0, 1, 2, 3],
                  forbidden_leaves: Union[list[int], int, None] = None
                  ) -> None:
    '''
    for every matrix which was generated: Load it, and use the
    pipeline on it. Generate a new Output-Object for every of them
    and sum up Runtimes etc.
    '''
    # Load the files
    filePaths = list(test_set.glob('*'))

    # Get overall number of used scenarios
    number_of_scenarios = len(filePaths)
    # TODO: Re-add progress bar?
    with Pool() as pool:
        # For every file, use the pipeline ~ loop it baby, loop it!
        # Eww, my CPU get's so bored by this ~ USE THE DAMN CORES!
        benchmark = Benchmark(work_package, forbidden_leaves)
        # Lazily construct our statistics
        statistics = pool.imap_unordered(benchmark, filePaths)
        # Reduce all the statistics into a single one and print that
        # We'll need to do this here to force lazy evaluation to proceed
        final_statistic = reduce(BenchmarkStatistics.add, statistics)
        final_statistic.pretty_print(number_of_scenarios, work_package)


def wp2benchmark(test_set: Path) -> None:
    benchmark_all(test_set, work_package=WORK_PACKAGE_2)


def wp31benchmark(test_set: Path) -> None:
    benchmark_all(test_set, work_package=WORK_PACKAGE_3, forbidden_leaves=[0, 1, 2])


def wp32benchmark(test_set: Path) -> None:
    benchmark_all(test_set,
                  work_package=WORK_PACKAGE_3,
                  forbidden_leaves=[0, 1, 2, 3])


def wp341benchmark(test_set: Path) -> None:
    benchmark_all(test_set, work_package=WORK_PACKAGE_3_4, forbidden_leaves=3)


def wp342benchmark(test_set: Path) -> None:
    benchmark_all(test_set, work_package=WORK_PACKAGE_3_4, forbidden_leaves=4)
