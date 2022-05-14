from ActiveLearner import ActiveLearner
import random

#Chooses NumberOfElements Elements randomly from unused trainingdata in basecorpus
class Random(ActiveLearner):
    def SelectData(self, NumberOfElements: int):
        return self.SelectRandomData(NumberOfElements)



