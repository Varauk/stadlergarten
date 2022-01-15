# stadlergarten
Repository for group Megaman, course: Graphentheorie WS2021

## Setup
- Requires a modified version of the package `erdbeermet` to be installed, see `erdbeermet/`(`python setup.py install`, based on original version 0.0.4). It requires the following packages
  - [Scipy](http://www.scipy.org/install.html) (`pip install scipy`)
  - [Matplotlib](https://matplotlib.org/) (`pip install matplotlib`)
  - [Numpy](https://numpy.org) (`pip install numpy`)
- [tqdm](https://pypi.org/project/tqdm/) (`pip install tqdm`)

## Running
- Main file: `praktikum/run.py`, available parameters are printed when running it
- Example usage:
  - `run.py -p 2 -s small`
  - `run.py -p all -s large --debug --writeResultsToFiles`
- Parameter `--workpackage/-p` allows selecting a workpackage (see `documents/Praktikumsbeschreibung_v1.5.pdf`) for details)
  - **2**: Runs the pipeline according to the description of WP2
  - **31**: Runs the pipeline according to WP3, third bullet point, using [0, 1, 2] as the first leaves
  - **32**: Runs the pipeline according to WP3, third bullet point, using [0, 1, 2, 3] as the first leaves
  - **331**: Runs the pipeline according to WP3, last bullet point, looking for a correctly identified subset of 3 leaves
  - **332**: Runs the pipeline according to WP3, last bullet point, looking for a correctly identified subset of 4 leaves
  - **41**: Runs the pipeline according to the description in WP4
  - **42**: Same as **41**, but uses the provided `_compute_deltas` method from the `erdbeermet` package to compute the spike lengths
  - **all**: Executes all of the above mentioned workpackages sequentially
- When using `-w` a subfolder `benchmarkOutput` will be created containing the benchmark results

## References
- [Erdbeermet](https://github.com/david-schaller/Erdbeermet)
- [Praktikumsunterlagen](http://silo.bioinf.uni-leipzig.de/GTPraktikumRMaps/)
