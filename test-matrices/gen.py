#!/usr/bin/env python3

from sys import argv, exit
import math
import random
from pathlib import Path
from typing import Final

from erdbeermet.simulation import simulate

PROP_CIRCULAR: Final[float] = 0.25
PROP_CLOCKLIKE: Final[float] = 0.25
DEFAULT_MIN_MATRIX_SIZE: Final[int] = 5
DEFAULT_MAX_MATRIX_SIZE: Final[int] = 10
USAGE = 'Usage: ./gen.py FOLDER NUMBER_OF_MATRICES [MIN_MATRIX_SIZE] [MAX_MATRIX_SIZE]'


def generate(folder: Path,
             count: int,
             min_size: int,
             max_size: int) -> None:
    # Really?
    if (count == 0):
        print('Done... Idiot...')
        exit()

    # How many digits does our count have?
    count_len = math.ceil(math.log(count, 10))
    size_len = math.ceil(math.log(max_size, 10))

    # Create the target directory if missing
    folder.mkdir(parents=True, exist_ok=True)

    # Repeat count times
    for idx in range(count):
        is_circular = random.uniform(0.0, 1.0) < PROP_CIRCULAR
        is_clocklike = random.uniform(0.0, 1.0) < PROP_CLOCKLIKE
        size = random.randint(min_size, max_size)

        filename = '{:0>{count_len}}-{:0>{size_len}}-{}{}.txt'.format(
            idx,
            size,
            'i' if is_circular else '-',
            'o' if is_clocklike else '-',
            count_len=count_len,
            size_len=size_len)
        print('Generating {}.. '.format(filename), end='')

        szenario = simulate(size, circular=is_circular, clocklike=is_clocklike)
        szenario.write_history(folder/filename)
        print('done')


def delete_old_hists(folder: Path) -> None:
    print('Removing old hist files.. ', end='')
    hists = folder.glob('*')
    for hist in hists:
        hist.unlink()
    print('done')


if __name__ == '__main__':
    if len(argv) < 3:
        print(USAGE)
        exit(1)
    folder = Path(argv[1])
    count = int(argv[2])

    min_size = int(argv[3]) if len(argv) > 3 else DEFAULT_MIN_MATRIX_SIZE
    max_size = int(argv[4]) if len(argv) > 4 else DEFAULT_MAX_MATRIX_SIZE
    delete_old_hists(folder)
    generate(folder=folder, count=count, min_size=min_size, max_size=max_size)
