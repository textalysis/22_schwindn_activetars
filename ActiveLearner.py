import copy, random
from flair.data import Subset, Dataset, _len_dataset, Corpus, Sentence
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

#Base class implementing methods and variables used by all active learning algorithms
class ActiveLearner:
    def __init__(
        self,
        corpus: Corpus,
        TARS: TARSClassifier,
        LabelType: str = 'class',
        task: str = "question classification",
        multilabel: bool = False,
        ):
        self.basecorpus = corpus
        self.TARS = TARS
        self.UsedIndices = []
        self.currentTrainCorpus = copy.deepcopy(self.basecorpus)
        self.LabelType = LabelType
        self.LabelDict = self.basecorpus.make_label_dictionary(label_type = self.LabelType)
        self.CorpusLabels = self.LabelDict.get_items()
        self.CorpusLabels.remove('<unk>')
        self.task = task
        self.multilabel = multilabel
        self.TARS.add_and_switch_to_new_task(task_name=self.task,
                                             label_dictionary=self.LabelDict,
                                             label_type=self.LabelType,
                                             )

    #Classifies a dataset
    def classifyCorpus(self, dataset: Dataset):
        sentences = [Sentence(data.to_plain_string()) for data in dataset]
        self.TARS.predict_zero_shot(sentences, self.CorpusLabels, multi_label=self.multilabel)
        return sentences

    #Evaluates the Model on the validation Dataset of the Corpus
    def evaluateModel(self):
        AccuratePredictions = 0
        TotalPredictions = 0
        PredictedSentences = self.classifyCorpus(self.basecorpus.dev)
        for sentence, data in zip(PredictedSentences, self.basecorpus.dev):
            maxlabel = [label for label in sentence.labels if
                        label.score == max([label.score for label in sentence.labels])]
            try:
                if maxlabel[0].value != data.labels[0].value or maxlabel == []:
                    TotalPredictions += 1
                else:
                    AccuratePredictions += 1
                    TotalPredictions += 1
            except:
                TotalPredictions += 1
        return AccuratePredictions / TotalPredictions

    #Downsamples the Corpus to only include data with given indices
    def downsampleCorpus(
        self,
        IndicesToKeep: list = [],
        DownsampleTrainSet = True,
        DownsampleDevSet = True,
        DownsampleTestSet = True,
        ):
        downsampledCorpus = copy.deepcopy(self.basecorpus)
        if DownsampleTrainSet and downsampledCorpus._train is not None:
            downsampledCorpus._train = self.splitDataset(downsampledCorpus._train, IndicesToKeep)
        if DownsampleDevSet and downsampledCorpus._train is not None:
            downsampledCorpus._dev = downsampledCorpus._downsample_to_proportion(downsampledCorpus._dev, 0.1*(len(IndicesToKeep))/_len_dataset(downsampledCorpus._dev))
        if DownsampleTestSet and downsampledCorpus._train is not None:
            downsampledCorpus._test = downsampledCorpus._downsample_to_proportion(downsampledCorpus._test, 0.1 * (len(IndicesToKeep)) / _len_dataset(downsampledCorpus._test))
        self.currentTrainCorpus = downsampledCorpus
        return downsampledCorpus

    #Splits a Dataset into two, First Dataset includes all data with indices in IndicesToKeep, second Dataset all the others
    def splitDataset(
        self,
        dataset: Dataset,
        IndicesToKeep: list = [] ,
        ):
        IndicesToKeep.sort()
        #IndicesToRemove = [i for i in range(len(dataset)) if i not in IndicesToKeep].sort()

        return Subset(dataset, IndicesToKeep)#, Subset(dataset, IndicesToRemove)

    #Trains the TARS model with the current Train Corpus, provided by the active learning strategy
    def trainTARS(self,path: str = 'resources/taggers/trec'):
        self.TARS.add_and_switch_to_new_task(task_name=self.task,
                                             label_dictionary=self.LabelDict,
                                             label_type=self.LabelType,
                                             )

        trainer = ModelTrainer(self.TARS, self.currentTrainCorpus)

        trainer.train(base_path=path,  # path to store the model artifacts
                      learning_rate=0.02,  # use very small learning rate
                      mini_batch_size=16,
                      mini_batch_chunk_size=4,  # optionally set this if transformer is too much for your machine
                      max_epochs=10,  # terminate after 10 epochs
                      )

        return self.TARS

    #Chooses NumberOfElements Elements randomly from unused trainingdata in basecorpus
    def SelectRandomData(self, NumberOfElements: int):
        selectableIndices = [i for i in list(range(len(self.basecorpus.train))) if i not in self.UsedIndices]
        randomIndices = random.sample(selectableIndices, NumberOfElements)
        self.UsedIndices.extend(randomIndices)
        return self.downsampleCorpus(IndicesToKeep = randomIndices)

    #Prints the data that is selected by the active learning strategy
    def printCurrentTrainingData(self):
        print('The current Trainingdata is:')
        print([trainingdata for trainingdata in self.currentTrainCorpus.train])





