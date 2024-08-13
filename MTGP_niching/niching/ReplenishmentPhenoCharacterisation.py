import MTGP_niching.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import MTGP_niching.sequencing as sequencing


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
            candidate_action = replenishment_data[1]
            quantity = sequencing.GP_evolve_S(state, self.referenceRule)
            index = 0
            min_dis = np.Infinity
            for i in range(len(candidate_action)):
                dis = np.abs(quantity - candidate_action[i])
                if dis < min_dis:
                    index = i
                    min_dis = dis
            self.decisions.append(candidate_action[index])

    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            replenishmentDecision = self.decisionSituations[i].clone()
            replenishment_data = replenishmentDecision.getData()
            state = replenishment_data[0]
            candidate_action = replenishment_data[1]
            quantity = sequencing.GP_evolve_S(state, rule)
            index = 0
            min_dis = np.Infinity
            for i in range(len(candidate_action)):
                dis = np.abs(quantity - candidate_action[i])
                if dis < min_dis:
                    index = i
                    min_dis = dis

            charlist.append(candidate_action[index])

        return charlist


