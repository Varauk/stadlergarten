#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
import logging
from typing import Final, List, Optional
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime

import pipeline
from pipeline import benchmark_all, BenchmarkStatistics
from workpackage import WorkPackage
from plotwhen import PlotWhen

TEST_SET_DIR: Final[Path] = Path('../test-matrices/')


def test_set_choices() -> List[str]:
    possible_files = TEST_SET_DIR.glob('*')
    directories = filter(Path.is_dir, possible_files)
    dir_names = [dir.name for dir in directories]
    return list(dir_names)


def parse_cli_arguments() -> Namespace:
    parser = ArgumentParser(description='Graph Theory Pipeline')
    parser.add_argument('--workpackage', '-p',
                        choices=['2', '31', '32', '331',
                                 '332', '41', '42', 'all'],
                        required=True,
                        help='Select work package to execute')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--test-set', '-s',
                        choices=test_set_choices(),
                        required=True)
    parser.add_argument('--plot-when', '-o',
                        choices=[var.value for var in PlotWhen],
                        help='When to plot the reconstructed R-steps',
                        default='never')
    parser.add_argument('--cores', '-j',
                        help='Number of cores to use',
                        default=None)
    parser.add_argument('--writeResultsToFiles', '-w', action='store_true')
    parser.add_argument('--seed',
                        help='RNG seed to use',
                        default=None)
    return parser.parse_args()


def setup_logging(enable_debug: bool) -> None:
    '''Setup logger and enable `info` output if requested'''
    level = logging.INFO if enable_debug else logging.WARN
    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    logfile = log_folder / Path(datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.log')
    logging.basicConfig(filename=logfile, encoding='utf-8', level=level)


def execute_workpackage(wp: WorkPackage,
                        test_set: Path,
                        rng_seed: Optional[int],
                        nr_of_cores: Optional[int],
                        plot_when: PlotWhen) -> BenchmarkStatistics:
    return benchmark_all(test_set=test_set,
                         work_package=wp,
                         rng_seed=rng_seed,
                         plot_when=plot_when,
                         forbidden_leaves=wp.get_forbidden_leaves(),
                         nr_of_cores=nr_of_cores)


def execute_workpackages(workpackages: List[WorkPackage],
                         test_set: Path,
                         rng_seed: Optional[int],
                         plot_when: PlotWhen,
                         nr_of_cores: Optional[int],
                         write_results: bool) -> None:
    '''Executes all of the given workpackages.
       Results will be written, if so desired.'''
    # If we have more than one workpackage,
    # do Stefan's timer thing
    if len(workpackages) > 1:
        start_time = timer()
    # Run every workpackage
    for wp in workpackages:
        statistics = execute_workpackage(wp=wp,
                                         test_set=test_set,
                                         rng_seed=rng_seed,
                                         plot_when=plot_when,
                                         nr_of_cores=nr_of_cores)
        # If requested, save the results
        if write_results:
            statistics.writeToFile(wp)
    # If there was more than one, evaluate the timing
    if len(workpackages) > 1:
        end_time = timer()
        total_runtime = end_time - start_time
        print(
            'Finished running all workpackage simulations on set '
            + str(test_set)
            + ' (took '
            + pipeline.pretty_time(total_runtime) + ')'
        )


def main() -> None:
    args = parse_cli_arguments()
    setup_logging(args.debug)
    # Select test set directory
    test_set = TEST_SET_DIR/args.test_set
    # Plot when?
    # Parse the raw value into an enum variant. This should be safe,
    # since argparse makes sure only allowed choices appear here
    plot_when = PlotWhen(args.plot_when)
    # Number of cores to use, `None` will just use all
    nr_of_cores = int(args.cores) if args.cores is not None else None
    # Parse the given workpackage string into
    # a list of workpackages to execute
    workpackages = WorkPackage.from_cli_arg(args.workpackage)
    # Execute selected workpackage
    execute_workpackages(workpackages=workpackages,
                         test_set=test_set,
                         rng_seed=args.seed,
                         plot_when=plot_when,
                         nr_of_cores=nr_of_cores,
                         write_results=args.writeResultsToFiles)


if __name__ == '__main__':
    main()
