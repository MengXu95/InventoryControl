import CCGP_niching_rental.niching.DecisionSituation as DecisionSituation

class RentalDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, data, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self, data)

    def clone(self):
        dataClone = []
        for op in self.data:
            dataClone.append(op)
        return RentalDecisionSituation(dataClone)


