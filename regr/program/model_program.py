from .program import LearningBasedProgram
from .model.pytorch import PoiModel, PoiModelToWorkWithLearnerWithLoss, SolverModel, SolverModelDictLoss, SolverListPOIModel
from .model.iml import IMLModel


class POIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, PoiModel, **kwargs)


class SolverPOIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, SolverModel, **kwargs)

### chen begin List POI
class SolverListPOIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, SolverListPOIModel, **kwargs)
### chen end
        

class SolverPOIDictLossProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, SolverModelDictLoss, **kwargs)


class IMLProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, IMLModel, **kwargs)


class POILossProgram(LearningBasedProgram):
    def __init__(self, graph, poi=None):
        super().__init__(graph, PoiModelToWorkWithLearnerWithLoss, poi=poi)
