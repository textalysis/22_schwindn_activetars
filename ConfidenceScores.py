from ActiveLearner import ActiveLearner
from flair.data import Sentence
import numpy as np

#Chooses NumberOfElements Elements according to Confidence Scores algorithm from unused trainingdata in basecorpus
class ConfidenceScores(ActiveLearner):
    def SelectData(self, NumberOfElements: int):
        self.TARS.eval()
        classifiedCorpus = [Sentence(data.to_plain_string()) for
                            data in self.basecorpus.train]
        self.TARS.predict_zero_shot(classifiedCorpus,
                                    self.CorpusLabels,
                                    multi_label=True)
        SentenceScoresAndIndex = []
        SentenceIndex = 0
        for sentence in classifiedCorpus:
            SumAllLabelScores = sum([label.score for label in sentence.labels])
            entropy = -sum(
                [(label.score / SumAllLabelScores) * np.log(label.score / SumAllLabelScores)
                 for label in sentence.labels])
            SentenceScoresAndIndex.append((entropy, SentenceIndex))
            SentenceIndex += 1
        SentenceScoresAndIndex_sorted = sorted(SentenceScoresAndIndex,
                                               key=lambda tup: -tup[0])
        SentenceScoresAndIndex_sorted = [i for i in SentenceScoresAndIndex_sorted
                                         if i[1] not in self.UsedIndices]
        Selections = SentenceScoresAndIndex_sorted[:NumberOfElements]
        SelectedIndices = [i[1] for i in Selections]
        self.UsedIndices.extend(SelectedIndices)
        self.TARS.train()
        self.downsampleCorpus(IndicesToKeep=self.UsedIndices)
        return self.downsampleCorpusEval(IndicesToKeep = self.UsedIndices)


