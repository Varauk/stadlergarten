
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

    # def __init__(self,
    #              classifiedAsRMap,
    #              classifiedMatchingFourLeaves,
    #              divergence,
    #              measuredRuntime,
    #              plotMatrix,
    #              plotSteps,
    #              plotBoxPlots):
    #    self.classifiedAsRMap = classifiedAsRMap
    #    self.classifiedMatchingFourLeaves = classifiedMatchingFourLeaves
    #    self.divergence = divergence
    #    self.measuredRuntime = measuredRuntime
    #    self.plotMatrix = plotMatrix
    #    self.plotSteps = plotSteps
    #    self.plotBoxPlots = plotBoxPlots

    def print(self):
        print("Statistics for this runtrough:\n")

        if self.classifiedAsRMap:
            print("* Was correctly recognized as an R-Map.\n")
        if self.classifiedMatchingFourLeaves:
            print("* The final 4-leaf-map matches the original 4-leaf-map!\n")

        print("* Count of diverging steps was {}.\n".format(self.divergence))

        if self.measuredRuntime != 0.0:
            print("* The task took {} seconds.\n".format(self.measuredRuntime))

        # Plotters
        if self.plotMatrix:
            pass
        if self.plotSteps:
            pass
        if self.plotMatrix:
            pass
