#!/usr/bin/env python3

import os
from sys import argv, exit
import math
import random
from pathlib import Path

from erdbeermet.simulation import simulate

PATH=Path('./hists')
PROP_CIRCULAR=0.25
PROP_CLOCKLIKE=0.25
DEFAULT_MIN_MATRIX_SIZE=4
DEFAULT_MAX_MATRIX_SIZE=25

def generate(count, min_size, max_size):
    # Really?
    if (count == 0):
        print('Done... Idiot...')
        exit()

    # How many digits does our count have?
    count_len = math.ceil(math.log(count, 10))
    size_len = math.ceil(math.log(max_size, 10))

    # Create the target directory if missing
    PATH.mkdir(parents=True, exist_ok=True)

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
            count_len = count_len,
            size_len = size_len)
        print('Generating {} ..'.format(filename), end='')

        szenario = simulate(size, circular=is_circular, clocklike=is_clocklike)
        szenario.write_history(PATH/filename)
        print('done')

if __name__ == '__main__':
    if len(argv) < 2:
        print('Usage: ./gen.py NUMBER_OF_MATRICES [MIN_MATRIX_SIZE] [MAX_MATRIX_SIZE]')
        exit(1)
    count = int(argv[1])
    min_size = argv[2] if len(argv) >= 3 else DEFAULT_MIN_MATRIX_SIZE
    max_size = argv[3] if len(argv) >= 4 else DEFAULT_MAX_MATRIX_SIZE
    generate(count=count, min_size=min_size, max_size=max_size)
