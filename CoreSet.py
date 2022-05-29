from flair.data import Sentence, Corpus
from ActiveLearner import ActiveLearner
import math
import numpy as np
from flair.embeddings import TokenEmbeddings
from flair.models import TARSClassifier

#Chooses NumberOfElements Elements according to Core Set algorithm from unused trainingdata in basecorpus
class CoreSet(ActiveLearner):
    def __init__(self, corpus: Corpus, TARS: TARSClassifier):
        super().__init__(corpus, TARS)
        self.DistanceMatrix = []
    def SelectData(self, NumberOfElements: int):
        PossibleTrainData = [data.to_plain_string() for data in self.basecorpus.train]
        PossibleTrainData = [Sentence(sentence) for sentence in PossibleTrainData]
        self.TARS.tars_embeddings.eval()
        self.TARS.tars_embeddings.embed(PossibleTrainData)
        self.TARS.tars_embeddings.train()

        if isinstance(self.TARS.tars_embeddings, TokenEmbeddings):
            encodings_np = [sentence[0].get_embedding().cpu().detach().numpy() for sentence in PossibleTrainData]
        else:
            encodings_np = [sentence.get_embedding().cpu().detach().numpy() for sentence in PossibleTrainData]
        if self.DistanceMatrix == []:
            distance = lambda embedding1, embedding2: math.sqrt(
                sum([(embedding1[i] - embedding2[i]) ** 2 for i in range(len(embedding1))]))
            self.distancematrix = distance
            print('Core Set: building distance matrix')
            DistanceMatrix = np.asarray(
                [[distance(embedding1, embedding2) for embedding2 in encodings_np] for embedding1 in encodings_np])
            self.DistanceMatrix = DistanceMatrix
        startPoint = min([index for index in range(len(self.basecorpus.train)) if index not in self.UsedIndices])
        SelectedIndices = self.KCenterGreedy(self.DistanceMatrix, startPoint, NumberOfElements)
        self.UsedIndices.extend(SelectedIndices)

        return self.downsampleCorpus(IndicesToKeep = SelectedIndices)

    def KCenterGreedy(self, DistanceMatrix, StartPoint, NumberOfElements):
        chosenDataPoints = [StartPoint]
        ValuesForIndices = {}
        while len(chosenDataPoints) < NumberOfElements:
            print(f'Core Set: {str(len(chosenDataPoints))} training data found')
            for j in [j for j in range(len(DistanceMatrix[0])) if (j not in chosenDataPoints) and (j not in self.UsedIndices)]:
                ValuesForIndices[min(DistanceMatrix[:, j].take(chosenDataPoints))] = j
            chosenDataPoints.append(ValuesForIndices[max(ValuesForIndices.keys())])
            ValuesForIndices = {}
        return chosenDataPoints