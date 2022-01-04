#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
from typing import Final
from pathlib import Path

import pipeline

TEST_SET_DIR: Final[Path] = Path('../test-matrices/')


def test_set_choices() -> list:
    possible_files = TEST_SET_DIR.glob('*')
    directories = filter(Path.is_dir, possible_files)
    dir_names = [dir.name for dir in directories]
    return list(dir_names)


def main():
    parser = ArgumentParser(description='Graph Theory Pipeline')
    parser.add_argument('--workpackage', '-p',
                        choices=['2', '31', '32', '341', '342'],
                        required=True,
                        help='Select work package to execute')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--test-set', '-s',
                        choices=test_set_choices(),
                        required=True)
    args = parser.parse_args()

    # Enable debugging if requested
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)

    # Select test set directory
    test_set = TEST_SET_DIR/args.test_set

    # Execute selected workpackage
    if args.workpackage == '2':
        pipeline.wp2benchmark(test_set)
    elif args.workpackage == '31':
        pipeline.wp31benchmark(test_set)
    elif args.workpackage == '32':
        pipeline.wp32benchmark(test_set)
    elif args.workpackage == '341':
        pipeline.wp341benchmark(test_set)
    elif args.workpackage == '342':
        pipeline.wp342benchmark(test_set)


if __name__ == '__main__':
    main()
