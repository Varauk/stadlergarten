#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
from typing import Final, List
from pathlib import Path
from timeit import default_timer as timer

import pipeline
from pipeline import PlotWhen

TEST_SET_DIR: Final[Path] = Path('../test-matrices/')


def test_set_choices() -> List[str]:
    possible_files = TEST_SET_DIR.glob('*')
    directories = filter(Path.is_dir, possible_files)
    dir_names = [dir.name for dir in directories]
    return list(dir_names)


def main() -> None:
    parser = ArgumentParser(description='Graph Theory Pipeline')
    parser.add_argument('--workpackage', '-p',
                        choices=['2', '31', '32', '331', '332', '41', '42', 'all'],
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
    args = parser.parse_args()

    # Enable debugging if requested
    level = logging.INFO if args.debug else logging.WARN
    logging.basicConfig(level=level)

    # Select test set directory
    test_set = TEST_SET_DIR/args.test_set
    # Plot when?
    # Parse the raw value into an enum variant. This should be safe,
    # since argparse makes sure only allowed choices appear here
    plot_when = PlotWhen(args.plot_when)
    # Number of cores to use, `None` will just use all
    nr_of_cores = int(args.cores) if args.cores is not None else None

    # Execute selected workpackage
    if args.workpackage == '2':
        pipeline.wp2benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '31':
        pipeline.wp31benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '32':
        pipeline.wp32benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '331':
        pipeline.wp331benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '332':
        pipeline.wp332benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '41':
        pipeline.wp41benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == '42':
        pipeline.wp42benchmark(test_set, plot_when, nr_of_cores)
    elif args.workpackage == 'all':
        startTime = timer()
        # TODO add possibility to save the results to text files? Maybe make the functions return the BenchmarkStatistics object?
        pipeline.wp2benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp31benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp32benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp331benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp332benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp41benchmark(test_set, plot_when, nr_of_cores)
        pipeline.wp42benchmark(test_set, plot_when, nr_of_cores)
        endTime = timer()
        overallRuntime = endTime - startTime
        print(f'Finished running all workpackage simulations on set {args.test_set}',
              f'(took {overallRuntime :.2f} seconds)')


if __name__ == '__main__':
    main()
