import MTGP_niching_rental_RFQ.niching.PhenoCharacterisation as PhenoCharacterisation
import numpy as np

import MTGP_niching_rental_RFQ.rental as rental


class RentalPhenoCharacterisation(PhenoCharacterisation.PhenoCharacterisation):
    def __init__(self, referenceRule, decisionSituations, **kwargs):
        PhenoCharacterisation.PhenoCharacterisation.__init__(self, referenceRule)
        self.decisionSituations = decisionSituations.getData()
        self.decisions = []
        self.calcReferenceIndexes()

    def calcReferenceIndexes(self):
        self.decisions = []
        for i in range(len(self.decisionSituations)):
            rentalDecision = self.decisionSituations[i].clone()
            rental_data = rentalDecision.getData()
            rental_state = rental_data[0]

            # for making rental decision and delete not enough rental choice, by xu meng 2024.12.2
            all_rental_priority = []
            for each_rental_state in rental_state:
                current_rental = each_rental_state[0]
                rental_capacity = each_rental_state[2]
                total_rental_requirement = each_rental_state[-1]
                if current_rental + rental_capacity < total_rental_requirement:
                    rental_priority = np.inf
                else:
                    rental_priority = rental.GP_evolve_rental(each_rental_state, self.referenceRule)
                all_rental_priority.append(rental_priority)
            # Get the index of the minimal value
            #todo: currently, only consider the highest priority value of rental, but true rental can be a set of top
            #high priority choice
            rental_decision = all_rental_priority.index(min(all_rental_priority))
            self.decisions.append(rental_decision)

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
            rentalDecision = self.decisionSituations[i].clone()
            rental_data = rentalDecision.getData()
            rental_state = rental_data[0]

            # for making rental decision and delete not enough rental choice, by xu meng 2024.12.2
            all_rental_priority = []
            for each_rental_state in rental_state:
                current_rental = each_rental_state[0]
                rental_capacity = each_rental_state[2]
                total_rental_requirement = each_rental_state[-1]
                if current_rental + rental_capacity < total_rental_requirement:
                    rental_priority = np.inf
                else:
                    rental_priority = rental.GP_evolve_rental(each_rental_state, rule)
                all_rental_priority.append(rental_priority)
            # Get the index of the minimal value
            rental_decision = all_rental_priority.index(min(all_rental_priority))
            charlist.append(rental_decision)

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


