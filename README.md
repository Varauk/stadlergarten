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
- When using `-w` a subfolder `benchmarkOutput` will be created containing the benchmark results

## References
- [Erdbeermet](https://github.com/david-schaller/Erdbeermet)
- [Praktikumsunterlagen](http://silo.bioinf.uni-leipzig.de/GTPraktikumRMaps/)
