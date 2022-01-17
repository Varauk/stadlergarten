# Imports from libraries
from tqdm.contrib.concurrent import process_map

# Erdbeermet
from erdbeermet.simulation import simulate, load
from erdbeermet.recognition import recognize
from erdbeermet.tools.Tree import TreeNode

# Python packages
from typing import Optional, Union, List, Tuple, Any
from timeit import default_timer as timer
from logging import info
import itertools
import random
from pathlib import Path
from functools import reduce
import zipfile
import os
from datetime import datetime
from shutil import copyfile

# Own classes
from output import Output
from workpackage import WorkPackage
from plotwhen import PlotWhen

History = List[Tuple[int, int, int, float, float]]


class BenchmarkStatistics:
    timer_start: float
    timer_end: Optional[float]
    numberOfRMaps: int
    numberOfMatchingFourLeafs: int
    sumOfDivergenceWithOrder: float
    sumOfDivergenceWithoutOrder: float

    # set by pretty_print after a benchmark
    correctly_classified: float
    prop_4_leaf: float
    divergence_ordered: float
    divergence_unordered: float
    total_runtime: float

    def __init__(self) -> None:
        self.timer_start = timer()
        self.timer_end = None
        self.numberOfRMaps = 0
        self.numberOfMatchingFourLeafs = 0
        self.sumOfDivergenceWithOrder = 0.0
        self.sumOfDivergenceWithoutOrder = 0.0

    def stop_timer(self) -> None:
        self.timer_end = timer()

    def add(left: 'BenchmarkStatistics',
            right: 'BenchmarkStatistics') -> 'BenchmarkStatistics':
        sum = BenchmarkStatistics()
        sum.timer_start = min(left.timer_start, right.timer_start)
        # Make sure our timers are not None, if they are,
        # I'll assume they'll run forever
        left_t_end_f = left.timer_end or float('inf')
        right_t_end_f = right.timer_end or float('inf')
        sum.timer_end = max(left_t_end_f, right_t_end_f)
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
        self.correctly_classified = self.numberOfRMaps / total_count
        self.prop_4_leaf = self.numberOfMatchingFourLeafs / total_count
        self.divergence_ordered = self.sumOfDivergenceWithOrder / total_count
        self.divergence_unordered = self.sumOfDivergenceWithoutOrder / total_count
        if isinstance(self.timer_end, float):
            self.total_runtime = self.timer_end - self.timer_start
        else:
            self.total_runtime = float('inf')

        prettyTime = pretty_time(self.total_runtime)

        print(f'''
  +--------------= Benchmark =---------------+
  |                  Workpackage: {work_package :>9}  |
  | Number of simulated matrices: {total_count :>9}  |
  |     Overall runtime measured: {prettyTime :>10} |
  |  Correctly classified R-Maps: {self.correctly_classified :>10.2%} |
  |    Proportion of 4-leaf-maps: {self.prop_4_leaf :>10.2%} |
  |  Avg. divergence   (ordered): {self.divergence_ordered :>10.2%} |
  |  Avg. divergence (unordered): {self.divergence_unordered :>10.2%} |
  +------------------------------------------+
        ''')

    def writeToFile(self, work_package: WorkPackage) -> None:
        os.makedirs('benchmarkOutput', exist_ok=True)
        filename = Path('benchmarkOutput') / Path('benchmark_' + str(work_package) + '.txt')
        with open(filename, 'w') as f:
            f.write('workpackage=' + str(work_package) + '\n')
            f.write('totalRuntime=' + str(self.total_runtime) + '\n')
            f.write('correctlyClassified=' + str(self.correctly_classified) + '\n')
            f.write('correctlyClassifiedRmaps=' + str(self.prop_4_leaf) + '\n')
            f.write('avgDivergenceOrdered=' + str(self.divergence_ordered) + '\n')
            f.write('avgDivergenceUnordered=' + str(self.divergence_unordered) + '\n')
            f.write('timestamp=' + str(datetime.now()))


class Benchmark:
    '''This is just a benchmark function with some state attached.
       Used to allow the easy usage of multiprocessing.Pool'''
    work_package: WorkPackage
    forbidden_leaves: Union[List[int], int, None]
    plot_when: PlotWhen
    rng_seed: Optional[int]

    def __init__(self,
                 plot_when: PlotWhen,
                 work_package: WorkPackage,
                 rng_seed: Optional[int],
                 forbidden_leaves: Union[List[int], int, None]) -> None:
        self.work_package = work_package
        self.forbidden_leaves = forbidden_leaves
        self.plot_when = plot_when
        self.rng_seed = rng_seed

    def __call__(self, path: Path) -> BenchmarkStatistics:
        # We need to reset the seed here, because we're in a new process?
        # This will be reset for every single path, which sucks, but
        # it could be worse, nothing here needs 'security', it's just numbers
        random.seed(self.rng_seed)
        # Load the corresponding matrix with a new Output Object
        scenario = load(filename=path)
        # Write here the Wrapper which shall guess the core leaves and
        # tries to avoid them in the recognition. Run this until you find
        # a valid R-Map.
        # scenario.N has the number of items which were generated.
        # So we need all subsets of N items with 3 respectively 4 leaves.
        # ForbiddenLeaves is an int at the end of WP3.4 and a list at WP3.3
        combinationsOfLeaves = expand_leaves(self.forbidden_leaves, scenario.N)
        stats = BenchmarkStatistics()
        
        # Need a marker for failed Recognitions even after all combinations were tested
        failedMarker = True
        
        # Rotate until you find a valid solution
        for combination in combinationsOfLeaves:
            info(f'Checked combination of core leaves: {combination}')
            # The first leaves must correspond to the ones which
            # are forbidden and therefore can't be deleted
            # by the recognition algorithm.
            first_leaves = combination

            # Enable Spike-Length Calculation and corresponding calculation form
            useSpikes = False
            useErdbeermetComputation = False
            if self.work_package == WorkPackage.WP4_1:
                useSpikes = True
            elif self.work_package == WorkPackage.WP4_2:
                useSpikes = True
                useErdbeermetComputation = True

            output = pipeline(size=scenario.N,
                              predefinedSimulationMatrix=scenario.D,
                              measureDivergence=True,
                              first_leaves=first_leaves,
                              history=scenario.history,
                              plot_when=self.plot_when,
                              forbidden_leaves=combination,
                              use_spike_length=useSpikes,
                              use_erdbeermet_computation=useErdbeermetComputation)

            # WP3 is special. We iterate through different combinations.
            # But only a successful combination is interesting in terms of
            # R-Steps.
            if (not output.classified_as_r_map
                and (self.work_package == WorkPackage.WP3_3_1
                     or self.work_package == WorkPackage.WP3_3_2)):
                # Here we have a combination which was NOT valid.
                # so the R-steps are not interesting and we should
                # not count them into divergence.
                continue
            else:
                # Use the values of the current Output Object
                # to modify overall values of benchmark
                if (output.classified_as_r_map):
                    stats.numberOfRMaps += 1
                    
                    # At this point we found a valid solution
                    failedMarker = False
                    
                if (output.classified_as_matching_four_leaves):
                    stats.numberOfMatchingFourLeafs += 1
                stats.sumOfDivergenceWithOrder += output.divergence_with_order
                stats.sumOfDivergenceWithoutOrder += output.divergence_without_order
                
                # When we are in WP3, we are finished with the first
                # valid R-Map, so break the loop here.
                if (self.work_package == WorkPackage.WP3_3_1
                   or self.work_package == WorkPackage.WP3_3_2):
                    break

        # check the failed marker and write it to a specific directory if we didn't found a valid solution.
        if failedMarker: 
            info(f"Failed: {path}")
            self.write_failed_recognition(path, self.work_package)
                
        # Stop the timer and return the results
        stats.stop_timer()
        return stats

    def write_failed_recognition(self, path: Path, wp: WorkPackage) -> None:
        # Create parent directory (failed/WPx)
        target_dir = Path('failed') / Path(str(wp))
        target_dir.mkdir(exist_ok=True, parents=True)
        # Prepend the target with 'failed/'
        target = target_dir / path.name
        copyfile(path, target)
        # Hänge die Info zum Workpackage an.
        with open(target, 'a') as f:
            f.write('\n\n')
            f.write(f'Recognition algorithm failed for workpackage {self.work_package}.')


def pipeline(history: History,
             plot_when: PlotWhen,
             first_leaves: List[int],
             size: int = 10,
             predefinedSimulationMatrix: None = None,
             measureDivergence: bool = False,
             forbidden_leaves: Optional[List[int]] = None,
             print_info: bool = False,
             use_spike_length: bool = False,
             use_erdbeermet_computation: bool = False) -> Output:
    simulationMatrix = None

    if (predefinedSimulationMatrix is None):
        circular = random.choice((True, False))
        clocklike = random.choice((True, False))
        # generate scenario
        scenario = simulate(size, circular=circular, clocklike=clocklike)
        simulationMatrix = scenario.D
        info(scenario.D)

    else:
        # use supplied matrix
        simulationMatrix = predefinedSimulationMatrix

    output = recognize_wrapper(
        simulationMatrix,
        plot_when=plot_when,
        print_info=print_info,
        measureDivergence=measureDivergence,
        history=history,
        first_leaves=first_leaves,
        forbidden_leaves=forbidden_leaves,
        use_spike_length=use_spike_length,
        use_erdbeermet_computation=use_erdbeermet_computation)

    # print single outputs if debug is enabled
    info(output)

    return output


def recognize_wrapper(D: List[int],
                      history: History,
                      plot_when: PlotWhen,
                      first_leaves: List[int],
                      print_info: bool = False,
                      measureDivergence: bool = False,
                      forbidden_leaves: Optional[List[int]] = None,
                      use_spike_length: bool = False,
                      use_erdbeermet_computation: bool = False) -> Output:
    # Create our output object this also starts the timer
    output = Output()

    # Shall recognize skip forbidden leaves?
    recognition_tree = recognize(
        D,
        first_candidate_only=True,
        print_info=print_info,
        B=forbidden_leaves,
        use_spike_length=use_spike_length,
        use_erdbeermet_computation=use_erdbeermet_computation)

    # Check: Was the simulated Matrix an R-Map?
    if recognition_tree.root.valid_ways > 0:
        output.classified_as_r_map = True
        # Display the tree if the user wants to see it
        if plot_when == PlotWhen.ALWAYS:
            recognition_tree.visualize()
    else:
        # If the above failed, display the tree unless we should not
        if plot_when != PlotWhen.NEVER:
            recognition_tree.visualize()
        # set output values and return.
        output.set_failed_state()
        return output

    info(f'Valid ways of the root-Node: {recognition_tree.root.valid_ways}')
    # Check: Match reconstructed leaves and orginal leaves?
    # build a set which contains all childs
    possible_node_set = []

    for current_node in recognition_tree.postorder():
        if current_node.n == 4 and current_node.valid_ways == 1:
            possible_node_set.append(current_node)

    # Choose random one of the nodes as stated in WP2
    choosen_node = random.choice(possible_node_set)
    info(f'Randomly choosen last node: {choosen_node.V} | Corresponding first leaves: {first_leaves}')
    # Check current list of chosen_node against passLeafes-list
    if set(first_leaves).issubset(set(choosen_node.V)):
        output.classified_as_matching_four_leaves = True
        info('Leaves matched!')

    # Check: Do the R-Steps from the reconstructed tree
    # diverge from the original R-Steps?
    if measureDivergence:
        (unordered_div, ordered_div) = measure_divergence(
            recognition_tree=recognition_tree,
            history=history,
            choosen_node=choosen_node)
        output.divergence_without_order = unordered_div
        output.divergence_with_order = ordered_div

    # Make sure to stop the output timer
    output.stop_timer()
    return output


def measure_divergence(recognition_tree: Any,
                       history: History,
                       choosen_node: TreeNode) -> Tuple[float, float]:
    '''Measure the divergence of the recognized tree.
       Returns a tuple of (unordered, ordered) divergence.'''
    reconstructed_r_steps = []

    for current_node in recognition_tree.postorder():
        # add all r_steps where the result was an r-map at the end
        if current_node.valid_ways == 0 or current_node.R_step is None:
            continue
        # special case if we are at the end and have one of the
        # possible candidates, we only want that one which was chosen
        # randomly before at the leaf matching.
        # So we will skip all others.
        if current_node.n == 4 and current_node != choosen_node:
            continue
        # Construct a new R-step which is comparable
        # to those in the history
        temp = (current_node.R_step[0],
                current_node.R_step[1],
                current_node.R_step[2],
                round(current_node.R_step[3], 6))
        reconstructed_r_steps.append(temp)

    # Now we need to check the reconstructed r-steps against
    # the original ones. Extract r-steps from history
    history_r_steps = []
    for entry in history[3:]:
        # We have to modifiy the values and sort x,y in the
        # same order (ascending) as the reconstructed r_steps are.
        # The last entry of alpha does not match on the last
        # few digits sometimes. So I restricted it to 6 digits.
        if entry[0] > entry[1]:
            new_tuple = (entry[1], entry[0], entry[2], round(1-entry[3], 6))
        else:
            new_tuple = (entry[0], entry[1], entry[2], round(entry[3], 6))

        history_r_steps.append(new_tuple)

    info(f'R-Steps from history:\n{str(history_r_steps)}')
    # Now compare them. We use intersection to find elements
    # that were contained in both.

    # Matching entrys with order
    match_ordered = 0
    for index in range(len(history_r_steps)):
        historical = history_r_steps[index]
        reconstructed = reconstructed_r_steps[index]
        if historical == reconstructed_r_steps[index]:
            match_ordered += 1
        info(f'Compare: {historical} and {reconstructed}')
    info(f'Match Counter is: {match_ordered}')
    # Matching entrys without order
    match_unordered = len(
        set(history_r_steps).intersection(set(reconstructed_r_steps)))

    # return the result as one minus the proportion of successful
    # reconstructed steps from all original steps. Care for cases with n=4.
    if len(history_r_steps) == 0:
        # Return maximum divergence
        return (1.0, 1.0)
    else:
        return (1 - (match_unordered / len(history_r_steps)),
                1 - (match_ordered / len(history_r_steps)))


def expand_leaves(leaves: Union[List[int], int, None],
                  count: int) -> List[List[int]]:
    ''' Convert given leaves into a list of list of leaves
        This is mainly used for the expansion of forbidden_leaves
    '''
    if type(leaves) is list:
        return [leaves]
    elif type(leaves) is int:
        possible_leaves = range(count)
        leav_combinations = itertools.combinations(possible_leaves, leaves)
        leav_lists = [list(tupl) for tupl in leav_combinations]
        # We reverse the list because the solution [0,1,2,3]
        # is always trivial and the first.
        # Other solutions are more interesting.
        return list(reversed(leav_lists))
    else:
        # The default combination is the list with all the initial leaves
        # We use this if None of the leaves are forbidden
        return [[0, 1, 2, 3]]


def expand_hists_file(filePaths: List[Path]) -> List[Path]:
    ''' Make sure the given filePaths are not just a single zip.
        If the given list is a single zip, we'll extract that
        and return a list of paths contained in the zip. '''
    if len(filePaths) > 2:
        # There's more than just a folder + the original zip!
        return filePaths

    zip_files = list(filter(zipfile.is_zipfile, filePaths))
    if len(zip_files) != 1:
        # No way there's more than one zip in this folder, somethings wrong!
        return filePaths

    # There is only one zip file, let's handle that
    zip = zip_files[0]
    # Extract into `original/path/to/file.zip.d/`
    extract_dir = zip.parent / (zip.name + '.d')
    # Have we extracted this file already?
    if extract_dir.exists() and extract_dir.is_dir():
        # Yes, we have!
        info("Hists are zipped, but we've already extracted them! YEAH!")
        return list(extract_dir.glob('*'))
    else:
        # No, we have not!
        # Let's do it then
        info("Hists are zipped, I'll extract them for you! One sec..")
        with zipfile.ZipFile(zip) as file:
            file.extractall(extract_dir)
        return list(extract_dir.glob('*'))


def benchmark_all(test_set: Path,
                  nr_of_cores: Optional[int],
                  plot_when: PlotWhen,
                  rng_seed: Optional[int],
                  work_package: WorkPackage,
                  forbidden_leaves: Union[List[int], int, None] = None
                  ) -> BenchmarkStatistics:
    '''
    for every matrix which was generated: Load it, and use the
    pipeline on it. Generate a new Output-Object for every of them
    and sum up Runtimes etc.
    '''
    # Reset the seed here once, to make sure this thread is behaving aswell
    random.seed(rng_seed)
    # Load the files
    filePaths = list(test_set.glob('*'))
    # Large test sets are compressed into a single zip file
    # Let's make sure that's unpacked before we start
    filePaths = expand_hists_file(filePaths)

    # Get overall number of used scenarios
    number_of_scenarios = len(filePaths)

    # Eww, my CPU get's so bored by this ~ USE THE DAMN CORES!
    benchmark = Benchmark(work_package=work_package,
                          rng_seed=rng_seed,
                          forbidden_leaves=forbidden_leaves,
                          plot_when=plot_when)
    # Construct our statistics on all cores
    statistics_iter = process_map(benchmark,
                                  filePaths,
                                  max_workers=nr_of_cores,
                                  chunksize=1)
    # Reduce all the statistics into a single one and print that
    final_statistic = reduce(BenchmarkStatistics.add, statistics_iter)
    final_statistic.pretty_print(number_of_scenarios, work_package)
    return final_statistic


def pretty_time(secondsFloat: float) -> str:
    seconds = int(secondsFloat)
    if (seconds == 0) :
        return '0s'
    intervals = (
        ('w', 604800),  # 60 * 60 * 24 * 7
        ('d', 86400),    # 60 * 60 * 24
        ('hrs', 3600),    # 60 * 60
        ('min', 60),
        ('s', 1),
    )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            result.append("{}{}".format(value, name))
    return ''.join(result[:2])  # + '.' + f'{secondsFloat :.2f}'.split('.')[1]
