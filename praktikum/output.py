
class Output:
    classifiedAsRMap = False
    classifiedMatchingFourLeaves = False
    divergence = 0.0
    measuredRuntime = 0.0
    plotMatrix = False
    plotSteps = False
    plotBoxPlots = False

    def __init__(self):
        pass

    def __str__(self):
        string = 'Statistics for this runtrough:\n'

        if self.classifiedAsRMap:
            string += '* Was correctly recognized as an R-Map.\n'
        if self.classifiedMatchingFourLeaves:
            string += '* The final 4-leaf-map match the original 4-leaf-map!\n'

        string = f'* Count of diverging steps was {self.divergence}.\n'

        if self.measuredRuntime != 0.0:
            string += f'* The task took {self.measuredRuntime} seconds.\n'

        # TODO: Plotters
        if self.plotMatrix:
            pass
        if self.plotSteps:
            pass
        if self.plotMatrix:
            pass

        return string

    @DeprecationWarning
    def print(self):
        print(self)
