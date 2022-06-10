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
        self.currentTrainCorpus = []
        self.LabelType = LabelType
        self.LabelDict = self.basecorpus.make_label_dictionary(label_type = self.LabelType)
        self.CorpusLabels = self.LabelDict.get_items()
        self.CorpusLabels.remove('<unk>')
        self.task = task
        self.multilabel = multilabel
        self.learning_rate = 0.02
        self.mini_batch_size = 16
        self.mini_batch_chunk_size = 4
        self.max_epochs = 20
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
        PredictedSentences = self.classifyCorpus(self.basecorpus.test)
        for sentence, data in zip(PredictedSentences, self.basecorpus.test):
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

        TARS = TARSClassifier.load('tars-base')

        TARS.add_and_switch_to_new_task(task_name=self.task,
                                        label_dictionary=self.LabelDict,
                                        label_type=self.LabelType,
                                        )

        trainer = ModelTrainer(TARS, self.currentTrainCorpus)

        trainer.train(base_path=path,  # path to store the model artifacts
                      learning_rate=self.learning_rate,  # use very small learning rate
                      mini_batch_size=self.mini_batch_size,
                      mini_batch_chunk_size=self.mini_batch_chunk_size,  # optionally set this if transformer is too much for your machine
                      max_epochs=self.max_epochs,  # terminate after 20 epochs
                      )
        self.TARS = TARS

        return self.TARS

    #Chooses NumberOfElements Elements randomly from unused trainingdata in basecorpus
    def SelectRandomData(self, NumberOfElements: int):
        selectableIndices = [i for i in list(range(len(self.basecorpus.train))) if i not in self.UsedIndices]
        randomIndices = random.sample(selectableIndices, NumberOfElements)
        self.UsedIndices.extend(randomIndices)
        return self.downsampleCorpus(IndicesToKeep = self.UsedIndices)

    #Prints the data that is selected by the active learning strategy
    def printCurrentTrainingData(self):
        print('The current Trainingdata is:')
        print([trainingdata for trainingdata in self.currentTrainCorpus.train])

    def setSeedSet(self):
        RandomSeedIndices_train = [1466, 3676, 4287, 346, 4520, 847, 2974, 1354, 1531, 2538, 2834, 2208, 1321, 4340, 2079, 3770, 3152, 4853, 2811,
         4109, 642, 90, 177, 3710, 3853, 2800, 3171, 4361, 3672, 2401, 705, 2410, 3225, 3172, 4341, 3810, 4108, 474,
         176, 1051, 814, 3821, 4023, 124, 2446, 451, 1223, 2203, 4442, 2372, 1201, 2703, 2312, 3021, 2753, 1200, 2347,
         3117, 3690, 3716, 3355, 261, 889, 3922, 1675, 3106, 411, 3167, 1401, 1383, 3227, 2326, 3562, 3104, 482, 3459,
         4805, 349, 1942, 3239, 4132, 3677, 4511, 3447, 1533, 2580, 833, 3753, 854, 1760, 676, 322, 1421, 1788, 1836,
         1546, 341, 311, 516, 2204]
        RandomSeedIndices_dev = [322, 529, 485, 73, 382, 440, 502, 131, 411, 460]
        RandomSeedIndices_test = [148, 246, 217, 220, 40, 269, 230, 479, 349, 66]
        downsampledCorpus = copy.deepcopy(self.basecorpus)
        downsampledCorpus._train = self.splitDataset(downsampledCorpus._train, RandomSeedIndices_train)
        downsampledCorpus._dev = self.splitDataset(downsampledCorpus._dev, RandomSeedIndices_dev)
        downsampledCorpus._test = self.splitDataset(downsampledCorpus._test, RandomSeedIndices_test)
        self.UsedIndices.extend(RandomSeedIndices_train)
        self.currentTrainCorpus = downsampledCorpus
        return downsampledCorpus









