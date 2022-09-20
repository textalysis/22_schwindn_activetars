from flair.data import Sentence, Corpus
from ActiveLearner import ActiveLearner
import math
import numpy as np
from flair.embeddings import TokenEmbeddings
from flair.models import TARSClassifier
import torch
from sklearn.cluster import KMeans
import random

#TODO safe encodings instead of calculating them every round
#Chooses NumberOfElements Elements according to Core Set algorithm from unused trainingdata in basecorpus
class CoreSet(ActiveLearner):
    def __init__(self, corpus: Corpus, TARS: TARSClassifier, LabelType : str = 'class' ,device: str = 'cpu', shuffle: bool = True, mode: str = 'kCenter'):
        super().__init__(corpus, TARS, shuffle=shuffle, LabelType = LabelType)
        self.DistanceMatrix = []
        self.device = device
        self.mode = mode
        self.Clusters = {}
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
            if self.mode == 'kCenter':
                startPoint = min([index for index in range(len(self.basecorpus.train)) if index not in self.UsedIndices])
                SelectedIndices = self.KCenterGreedy(self.DistanceMatrix, startPoint, NumberOfElements)
            if self.mode == 'kMeans':
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
        if self.Clusters == {}:
            EmbeddedVectors = EmbeddedVectors.cpu().numpy()
            kmeans = KMeans(n_clusters=NumberOfElements).fit(EmbeddedVectors)
            for number, vector in enumerate(EmbeddedVectors):
                if kmeans.labels_[number] in self.Clusters.keys():
                    self.Clusters[kmeans.labels_[number]].append(number)
                else:
                    self.Clusters[kmeans.labels_[number]] = [number]
        chosenDataPoints = []
        for cluster in self.Clusters.values():
            if len([element for element in cluster if element not in self.UsedIndices]) > 1:
                DatapointsToChooseFrom = [e for e in cluster if e not in self.UsedIndices]
                chosenDataPoints.append(random.choice(DatapointsToChooseFrom))
            else:
                chosenDataPoints.append(random.choice(cluster))

        return chosenDataPoints

