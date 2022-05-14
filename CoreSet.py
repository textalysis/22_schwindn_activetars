from flair.data import Sentence
from ActiveLearner import ActiveLearner
import math
import numpy as np
from flair.embeddings import TokenEmbeddings

#Chooses NumberOfElements Elements according to Core Set algorithm from unused trainingdata in basecorpus
class CoreSet(ActiveLearner):
    def SelectData(self, NumberOfElements: int): #TODO dont choose already used indices
        PossibleTrainData = [self.basecorpus.train[i].to_plain_string() for i in range(len(self.basecorpus.train)) if i not in self.UsedIndices]
        PossibleTrainData = [Sentence(sentence) for sentence in PossibleTrainData]
        self.TARS.tars_embeddings.eval()
        self.TARS.tars_embeddings.embed(PossibleTrainData)
        self.TARS.tars_embeddings.train()

        if isinstance(self.TARS.tars_embeddings, TokenEmbeddings):
            encodings_np = [sentence[0].get_embedding().cpu().detach().numpy() for sentence in PossibleTrainData]
        else:
            encodings_np = [sentence.get_embedding().cpu().detach().numpy() for sentence in PossibleTrainData]

        distance = lambda embedding1, embedding2: math.sqrt(
            sum([(embedding1[i] - embedding2[i]) ** 2 for i in range(len(embedding1))]))
        DistanceMatrix = np.asarray(
            [[distance(embedding1, embedding2) for embedding2 in encodings_np] for embedding1 in encodings_np])
        SelectedIndices = self.KCenterGreedy(DistanceMatrix, 0, NumberOfElements)
        self.UsedIndices.extend(SelectedIndices)

        return self.downsampleCorpus(IndicesToKeep = SelectedIndices)

    def KCenterGreedy(self, DistanceMatrix, StartPoint, NumberOfElements):
        chosenDataPoints = [StartPoint]
        ValuesForIndices = {}
        while len(chosenDataPoints) < NumberOfElements:
            for j in [j for j in range(len(DistanceMatrix[0])) if j not in chosenDataPoints]:
                ValuesForIndices[min(DistanceMatrix[:, j].take(chosenDataPoints))] = j
            chosenDataPoints.append(ValuesForIndices[max(ValuesForIndices.keys())])
            ValuesForIndices = {}
        return chosenDataPoints