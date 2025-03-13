import MTGP_niching_rental_RFQ_price.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import MTGP_niching_rental_RFQ_price.replenishment as replenishment
from MTGP_niching_rental_RFQ_price import logistic_util


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
                quantity = round(replenishment.GP_evolve_S(state_retailer, self.referenceRule),2)

                # Strategy 2 (sigmoid): constrain the replenishment quantity to [0, production_capacity]
                # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
                #production_capacity = state_retailer[4]
                capacity = state_retailer[3]
                upbound_replenishment_quantity = capacity * 3
                if quantity > upbound_replenishment_quantity or quantity < 0:
                    quantity = logistic_util.logistic_scale_and_shift(quantity, 0,
                                                                                    upbound_replenishment_quantity)
                # print("replenishment_quantity after sigmoid: ", replenishment_quantity)

                self.decisions.append(quantity)

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
                quantity = round(replenishment.GP_evolve_S(state_retailer, rule),2)

                # Strategy 2 (sigmoid): constrain the replenishment quantity to [0, production_capacity]
                # Strategy 2: performs better than Strategy 1 based on one run with popsize 200
                production_capacity = state_retailer[4]
                capacity = state_retailer[3]
                upbound_replenishment_quantity = capacity * 3
                if quantity > upbound_replenishment_quantity or quantity < 0:
                    quantity = logistic_util.logistic_scale_and_shift(quantity, 0,
                                                                      upbound_replenishment_quantity)

                charlist.append(quantity)


        return charlist


