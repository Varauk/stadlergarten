
class Output:
    classifiedAsRMap = False
    classifiedMatchingFourLeaves = False
    divergenceWithoutOrder = 0.0
    divergenceWithOrder = 0.0
    # measuredRuntime = 0.0
    plotMatrix = False
    plotSteps = False
    plotBoxPlots = False

    def __init__(self) -> None:
        pass

    def __str__(self) -> str:
        string = 'Statistics for this runtrough:\n'

        if self.classifiedAsRMap:
            string += '* Was correctly recognized as an R-Map.\n'
        if self.classifiedMatchingFourLeaves:
            string += '* The final 4-leaf-map match the original 4-leaf-map!\n'

        string += ('* Proportion of diverging ' +
                   f'steps with order was {self.divergenceWithOrder :.2%}.\n')

        string += ('* Proportion of diverging steps ' +
                   f'without order was {self.divergenceWithoutOrder :.2%}.\n')

        # if self.measuredRuntime != 0.0:
        #     string += f'* The task took {self.measuredRuntime} seconds.\n'

        # TODO: Plotters
        if self.plotMatrix:
            pass
        if self.plotSteps:
            pass
        if self.plotMatrix:
            pass

        return string

    @DeprecationWarning
    def print(self) -> None:
        print(self)
