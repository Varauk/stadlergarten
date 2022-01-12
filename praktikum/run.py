#!/usr/bin/env python3

from argparse import ArgumentParser, Namespace
import logging
from typing import Final, List, Optional
from pathlib import Path
from timeit import default_timer as timer
from datetime import datetime

import pipeline
from pipeline import PlotWhen

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
    return parser.parse_args()


def setup_logging(enable_debug: bool) -> None:
    '''Setup logger and enable `info` output if requested'''
    level = logging.INFO if enable_debug else logging.WARN
    log_folder = Path('logs')
    log_folder.mkdir(exist_ok=True)
    logfile = log_folder / Path(str(datetime.now()) + '.log')
    logging.basicConfig(filename=logfile, encoding='utf-8', level=level)


def execute_workpackages(wp: str,
                         test_set: Path,
                         plot_when: PlotWhen,
                         nr_of_cores: Optional[int],
                         write_results: bool) -> None:
    if wp == '2':
        statisticwp2 = pipeline.wp2benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp2.writeToFile('2')
    elif wp == '31':
        statisticwp31 = pipeline.wp31benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp31.writeToFile('2')
    elif wp == '32':
        statisticwp32 = pipeline.wp32benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp32.writeToFile('2')
    elif wp == '331':
        statisticwp331 = pipeline.wp331benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp331.writeToFile('2')
    elif wp == '332':
        statisticwp332 = pipeline.wp332benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp332.writeToFile('2')
    elif wp == '41':
        statisticwp41 = pipeline.wp41benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp41.writeToFile('2')
    elif wp == '42':
        statisticwp42 = pipeline.wp42benchmark(test_set, plot_when, nr_of_cores)
        if (write_results):
            statisticwp42.writeToFile('2')
    elif wp == 'all':
        startTime = timer()
        # TODO add possibility to save the results to text files?
        # Maybe make the functions return the BenchmarkStatistics object?
        statisticwp2 = pipeline.wp2benchmark(test_set, plot_when, nr_of_cores)
        statisticwp31 = pipeline.wp31benchmark(test_set, plot_when, nr_of_cores)
        statisticwp32 = pipeline.wp32benchmark(test_set, plot_when, nr_of_cores)
        statisticwp331 = pipeline.wp331benchmark(test_set, plot_when, nr_of_cores)
        statisticwp332 = pipeline.wp332benchmark(test_set, plot_when, nr_of_cores)
        statisticwp41 = pipeline.wp41benchmark(test_set, plot_when, nr_of_cores)
        statisticwp42 = pipeline.wp42benchmark(test_set, plot_when, nr_of_cores)

        endTime = timer()
        overallRuntime = endTime - startTime

        print(
            'Finished running all workpackage simulations on set '
            + str(test_set)
            + ' (took '
            + pipeline.pretty_time(overallRuntime) + ')'
        )

        if (write_results):
            print('Writing output to files...')
            statisticwp2.writeToFile('2')
            statisticwp31.writeToFile('31')
            statisticwp32.writeToFile('32')
            statisticwp331.writeToFile('331')
            statisticwp332.writeToFile('332')
            statisticwp41.writeToFile('41')
            statisticwp42.writeToFile('42')
            print('Done!')


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
    # Execute selected workpackage
    execute_workpackages(wp=args.workpackage,
                         test_set=test_set,
                         plot_when=plot_when,
                         nr_of_cores=nr_of_cores,
                         write_results=args.writeResultsToFiles)


if __name__ == '__main__':
    main()
