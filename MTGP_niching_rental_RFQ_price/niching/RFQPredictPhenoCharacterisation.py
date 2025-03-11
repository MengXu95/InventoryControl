import MTGP_niching_rental_RFQ_price.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np
import MTGP_niching_rental_RFQ_price.RFQ_price_predict as RFQPredict
from MTGP_niching_rental_RFQ_price import logistic_util


class RFQPredictPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, demand_level = 100, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations.getData()
        self.decisions = []
        self.demand_level = demand_level
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.decisions = []
        for i in range(len(self.decisionSituations)):
            RFQ_PredictDecision = self.decisionSituations[i].clone()
            RFQ_Predict_data = RFQ_PredictDecision.getData()
            state = RFQ_Predict_data[0]
            for state_retailer in state:
                price = round(RFQPredict.GP_evolve_RFQ_predict(state_retailer, self.referenceRule),2)
                upbound_support_price = self.demand_level * 5

                if price <= 0 or price > upbound_support_price:
                    quantity = logistic_util.logistic_scale_and_shift(price, 0, upbound_support_price)

                self.decisions.append(price)


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
                price = round(RFQPredict.GP_evolve_RFQ_predict(state_retailer, self.referenceRule), 2)
                upbound_support_price = self.demand_level * 5

                if price <= 0 or price > upbound_support_price:
                    quantity = logistic_util.logistic_scale_and_shift(price, 0, upbound_support_price)

                charlist.append(price)
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


