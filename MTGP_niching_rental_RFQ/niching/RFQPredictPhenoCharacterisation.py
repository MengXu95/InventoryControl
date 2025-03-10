import MTGP_niching_rental_RFQ.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import MTGP_niching_rental_RFQ.RFQ_predict as RFQPredict
from MTGP_niching_rental_RFQ import logistic_util


class RFQPredictPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, support_level = 100, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations.getData()
        self.decisions = []
        self.support_level = support_level
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.decisions = []
        for i in range(len(self.decisionSituations)):
            RFQ_PredictDecision = self.decisionSituations[i].clone()
            RFQ_Predict_data = RFQ_PredictDecision.getData()
            state = RFQ_Predict_data[0]
            for state_retailer in state:
                quantity = round(RFQPredict.GP_evolve_RFQ_predict(state_retailer, self.referenceRule),2)
                upbound_support_quantity = self.support_level * 2

                if quantity <= 0 or quantity > upbound_support_quantity:
                    quantity = logistic_util.logistic_scale_and_shift(quantity, 0,
                                                                      upbound_support_quantity)

                self.decisions.append(quantity)


    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        charlist = []

        for i in range(len(self.decisionSituations)):
            RFQ_PredictDecision = self.decisionSituations[i].clone()
            RFQ_Predict_data = RFQ_PredictDecision.getData()
            state = RFQ_Predict_data[0]
            for state_retailer in state:
                quantity = round(RFQPredict.GP_evolve_RFQ_predict(state_retailer, rule),2)

                upbound_support_quantity = self.support_level * 2

                if quantity <= 0 or quantity > upbound_support_quantity:
                    quantity = logistic_util.logistic_scale_and_shift(quantity, 0,
                                                                      upbound_support_quantity)

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


