import CCGP_niching.niching.DecisionSituation as DecisionSituation

class TransshipmentDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, data, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self, data)

    def clone(self):
        dataClone = []
        for op in self.data:
            dataClone.append(op)
        return TransshipmentDecisionSituation(dataClone)


