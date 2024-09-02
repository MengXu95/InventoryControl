import CCGP_niching.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import CCGP_niching.transshipment as transshipment


class TransshipmentPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations.getData()
        self.decisions = []
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.decisions = []
        for i in range(len(self.decisionSituations)):
            transshipmentDecision = self.decisionSituations[i].clone()
            transshipment_data = transshipmentDecision.getData()
            state = transshipment_data[0]
            for state_retailer_pair in state:
                quantity = round(transshipment.GP_evolve_R(state_retailer_pair, self.referenceRule),2)
                self.decisions.append(quantity)
            # the following is the original with candidate selection
            # candidate_action = transshipment_data[1]
            # quantity = transshipment.GP_evolve_R(state, self.referenceRule)
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
            transshipmentDecision = self.decisionSituations[i].clone()
            transshipment_data = transshipmentDecision.getData()
            state = transshipment_data[0]
            for state_retailer_pair in state:
                quantity = round(transshipment.GP_evolve_R(state_retailer_pair, rule),2)
                charlist.append(quantity)
            # the following is the original with candidate selection
            # candidate_action = transshipment_data[1]
            # quantity = transshipment.GP_evolve_R(state, rule)
            # index = 0
            # min_dis = np.Infinity
            # for i in range(len(candidate_action)):
            #     dis = np.abs(quantity - candidate_action[i])
            #     if dis < min_dis:
            #         index = i
            #         min_dis = dis
            #
            # charlist.append(candidate_action[index])

        return charlist


