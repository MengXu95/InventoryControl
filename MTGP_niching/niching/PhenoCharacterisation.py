import numpy as np
class PhenoCharacterisation:
    def __init__(self, referenceRule, **kwargs):
        self.referenceRule = referenceRule

    def calcReferenceIndexes(self):
        pass

    def setReferenceRule(self, rule):
        self.referenceRule = rule
        self.calcReferenceIndexes()

    def characterise(self, rule):
        pass

    def distance(self, charList1, charList2):
        distance = 0.0
        for i in range(len(charList1)):
            diff = charList1[i] - charList2[i]
            distance += diff * diff
        return np.sqrt(distance)


