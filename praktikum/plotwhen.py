from enum import Enum


class PlotWhen(Enum):
    NEVER = 'never'
    ON_ERR = 'on-err'
    ALWAYS = 'always'
