import MTGP_niching.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import MTGP_niching.replenishment as replenishment


class ReplenishmentPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations.getData()
        self.decisions = []
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.decisions = []
        for i in range(len(self.decisionSituations)):
            replenishmentDecision = self.decisionSituations[i].clone()
            replenishment_data = replenishmentDecision.getData()
            state = replenishment_data[0]
            for state_retailer in state:
                quantity = round(replenishment.GP_evolve_S(state_retailer, self.referenceRule))
                self.decisions.append(quantity)
            # the following is the original with candidate selection
            # candidate_action = replenishment_data[1]
            # quantity = replenishment.GP_evolve_S(state, self.referenceRule)
            # index = 0
            # min_dis = np.Infinity
            # for i in range(len(candidate_action)):
            #     dis = np.abs(quantity - candidate_action[i])
            #     if dis < min_dis:
            #         index = i
            #         min_dis = dis
            # self.decisions.append(candidate_action[index])

    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            replenishmentDecision = self.decisionSituations[i].clone()
            replenishment_data = replenishmentDecision.getData()
            state = replenishment_data[0]
            for state_retailer in state:
                quantity = round(replenishment.GP_evolve_S(state_retailer, rule))
                charlist.append(quantity)
            # the following is the original with candidate selection
            # candidate_action = replenishment_data[1]
            # quantity = replenishment.GP_evolve_S(state, rule)
            # index = 0
            # min_dis = np.Infinity
            # for i in range(len(candidate_action)):
            #     dis = np.abs(quantity - candidate_action[i])
            #     if dis < min_dis:
            #         index = i
            #         min_dis = dis
            # charlist.append(candidate_action[index])

        return charlist


