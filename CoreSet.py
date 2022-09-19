from flair.data import Sentence, Corpus
from ActiveLearner import ActiveLearner
import math
import numpy as np
from flair.embeddings import TokenEmbeddings
from flair.models import TARSClassifier
import torch
from sklearn.cluster import KMeans
import random

#Chooses NumberOfElements Elements according to Core Set algorithm from unused trainingdata in basecorpus
class CoreSet(ActiveLearner):
    def __init__(self, corpus: Corpus, TARS: TARSClassifier, LabelType : str = 'class' ,device: str = 'cpu', shuffle: bool = True, mode: str = 'kCenter'):
        super().__init__(corpus, TARS, shuffle=shuffle, LabelType = LabelType)
        self.DistanceMatrix = []
        self.device = device
        self.mode = mode
    def SelectData(self, NumberOfElements: int):
        with torch.no_grad():
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
            if self.DistanceMatrix == [] and self.mode == 'kCenter':
                self.DistanceMatrix = torch.cdist(encodings_np, encodings_np)
                startPoint = min([index for index in range(len(self.basecorpus.train)) if index not in self.UsedIndices])
                SelectedIndices = self.KCenterGreedy(self.DistanceMatrix, startPoint, NumberOfElements)
            elif self.mode == 'kMeans':
                SelectedIndices = self.kMeans(encodings_np, NumberOfElements)
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

    def kMeans(self, EmbeddedVectors, NumberOfElements):
        EmbeddedVectors = EmbeddedVectors.cpu().numpy()
        kmeans = KMeans(n_clusters=NumberOfElements).fit(EmbeddedVectors)
        Clusters = {}
        for number, vector in enumerate(EmbeddedVectors):
            if kmeans.labels_[number] in Clusters.keys():
                Clusters[kmeans.labels_[number]].append(number)
            else:
                Clusters[kmeans.labels_[number]] = [number]
        chosenDataPoints = []
        for cluster in Clusters.values():
            chosenDataPoints.append(random.choice(cluster))

        return chosenDataPoints

