import MTGP_niching_rental_RFQ_price.niching.DecisionSituation as DecisionSituation

class ReplenishmentDecisionSituation(DecisionSituation.DecisionSituation):
    def __init__(self, data, **kwargs):
        DecisionSituation.DecisionSituation.__init__(self, data)

    def clone(self):
        dataClone = []
        for op in self.data:
            dataClone.append(op)
        return ReplenishmentDecisionSituation(dataClone)


