from flair.data import Sentence, Corpus
from ActiveLearner import ActiveLearner
import math
import numpy as np
from flair.embeddings import TokenEmbeddings
from flair.models import TARSClassifier
import torch

#Chooses NumberOfElements Elements according to Core Set algorithm from unused trainingdata in basecorpus
class CoreSet(ActiveLearner):
    def __init__(self, corpus: Corpus, TARS: TARSClassifier, LabelType : str = 'class' ,device: str = 'cpu', shuffle: bool = True):
        super().__init__(corpus, TARS, shuffle=shuffle, LabelType = LabelType)
        self.DistanceMatrix = []
        self.device = device
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
        print(len(encodings_np))
        print(len(encodings_np[0]))
        encodings_np = torch.tensor(encodings_np,device = self.device)
        if self.DistanceMatrix == []:
            self.DistanceMatrix = torch.cdist(encodings_np, encodings_np)
        startPoint = min([index for index in range(len(self.basecorpus.train)) if index not in self.UsedIndices])
        SelectedIndices = self.KCenterGreedy(self.DistanceMatrix, startPoint, NumberOfElements)
        self.UsedIndices.extend(SelectedIndices)
        self.downsampleCorpus(IndicesToKeep=self.UsedIndices)
        return self.downsampleCorpusEval(IndicesToKeep=SelectedIndices)

    def KCenterGreedy(self, DistanceMatrix, StartPoint, NumberOfElements):
        chosenDataPoints = [StartPoint]
        ValuesForIndices = {}
        while len(chosenDataPoints) < NumberOfElements:
            print(f'Core Set: {str(len(chosenDataPoints))} training data found')
            for j in [j for j in range(len(DistanceMatrix[0])) if (j not in chosenDataPoints) and (j not in self.UsedIndices)]:
                ValuesForIndices[torch.min(DistanceMatrix[:, j].take(torch.tensor(chosenDataPoints + self.UsedIndices, device = self.device)))] = j #just an experiment
            chosenDataPoints.append(ValuesForIndices[max(ValuesForIndices.keys())])
            ValuesForIndices = {}
        return chosenDataPoints