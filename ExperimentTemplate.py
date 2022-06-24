import flair, torch
import flair.datasets
from flair.models import TARSClassifier
from Definitions import label_name_map_50,label_name_map_stackoverflow
import Random as Rand
import ConfidenceScores as Conf
import ExpectedGradientLength as Expe
import CoreSet as Core
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

#flair.set_seed(100)
filename_results = 'results_CoreSet_Stackoverflow_seedset_2.txt'
filename_model = 'resources/taggers/CoreSet1'
filename_model2 = 'resources/taggers/CoreSet1'
device = 'cuda:1'
SeedSet = True
shuffle = True

Exp = 3  #1,2 oder 3

def write(name, contents):#, alg1, alg2lol, alg2):
    with open(filename_results, 'w', encoding='utf-8') as f:
        f.write('\n'.join(contents))
        #f.write('\n'+'Alg1 TrainDATA:')
        #for data in alg1.currentTrainCorpus.train:
        #    f.write('\n'+str(data))
        #if alg2lol:
        #    f.write('\n' + 'Alg2 TrainDATA:')
        #    for data in alg2.currentTrainCorpus.train:
        #        f.write('\n' + str(data))

flair.device = torch.device(device)

lines = []
#Initialize Models
if Exp == 1:
    TARS_Random = TARSClassifier.load('tars-base')
    TARS_ConfidenceScores = TARSClassifier.load('tars-base')
    alg2lol = True
elif Exp == 2:
    TARS_ExpectedGradientLength = TARSClassifier.load('tars-base')
    alg2lol = False
elif Exp == 3:
    TARS_CoreSet = TARSClassifier.load('tars-base')
    alg2lol = False

#Initialize Corpora
sample_value = 1
if Exp == 1:
    #TREC_Random = flair.datasets.TREC_50(label_name_map=label_name_map_50)#.downsample(sample_value)
    #TREC_ConfidenceScores = flair.datasets.TREC_50(label_name_map=label_name_map_50)#.downsample(sample_value)
    TREC_Random: Corpus = ClassificationCorpus('/vol/fob-vol7/mi19/schwindn/.flair/datasets/stackoverflow',
                                                 test_file='test.txt',
                                                 dev_file='dev.txt',
                                                 train_file='train.txt',
                                                 label_type='topic',
                                                 label_name_map=label_name_map_stackoverflow,
                                                 )
    TREC_ConfidenceScores: Corpus = ClassificationCorpus('/vol/fob-vol7/mi19/schwindn/.flair/datasets/stackoverflow',
                                               test_file='test.txt',
                                               dev_file='dev.txt',
                                               train_file='train.txt',
                                               label_type='topic',
                                               label_name_map=label_name_map_stackoverflow,
                                               )
elif Exp ==2:
    #TREC_ExpectedGradientLength = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
    TREC_ExpectedGradientLength: Corpus = ClassificationCorpus('/vol/fob-vol7/mi19/schwindn/.flair/datasets/stackoverflow',
                                               test_file='test.txt',
                                               dev_file='dev.txt',
                                               train_file='train.txt',
                                               label_type='topic',
                                               label_name_map=label_name_map_stackoverflow,
                                               )
elif Exp == 3:
    #TREC_CoreSet = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
    TREC_CoreSet: Corpus = ClassificationCorpus('/vol/fob-vol7/mi19/schwindn/.flair/datasets/stackoverflow',
                                               test_file='test.txt',
                                               dev_file='dev.txt',
                                               train_file='train.txt',
                                               label_type='topic',
                                               label_name_map=label_name_map_stackoverflow,
                                               )
#Initialize ActiveLearners
if Exp == 1:
    Random = Rand.Random(corpus = TREC_Random, TARS = TARS_Random, shuffle = shuffle, LabelType = 'topic')
    ConfidenceScores = Conf.ConfidenceScores(corpus = TREC_ConfidenceScores, TARS = TARS_ConfidenceScores, shuffle = shuffle,LabelType = 'topic')
elif Exp == 2:
    ExpectedGradientLength = Expe.ExpectedGradientLength(corpus = TREC_ExpectedGradientLength, TARS = TARS_ExpectedGradientLength, shuffle = shuffle,LabelType = 'topic')
elif Exp == 3:
    CoreSet = Core.CoreSet(corpus = TREC_CoreSet, TARS = TARS_CoreSet, device = device, shuffle = shuffle,LabelType = 'topic')

TrainSetSize = 50
if Exp == 1:
    BaseAccuracy1 = Random.evaluateModel()
    RandomAccuracy_seed = [BaseAccuracy1]
    Training_Data_Random = []
    BaseAccuracy2 = ConfidenceScores.evaluateModel()
    ConfidenceScoresAccuracy_seed = [BaseAccuracy2]
    Training_Data_ConfidenceScores = []
elif Exp == 2:
    BaseAccuracy = ExpectedGradientLength.evaluateModel()
    ExpectedGradientLengthAccuracy_seed = [BaseAccuracy]
    Training_Data_ExpectedGradientLength = []
elif Exp == 3:
    BaseAccuracy = CoreSet.evaluateModel()
    CoreSetAccuracy_seed = [BaseAccuracy]
    Training_Data_CoreSet = []


lines.append('Initialized Models:')
if Exp == 1:
    lines.append('Random Accuracy:')
    lines.append(', '.join(str(e) for e in RandomAccuracy_seed))
    lines.append('ConfidenceScores Accuracy:')
    lines.append(', '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
elif Exp == 2:
    lines.append('ExpectedGradientLength Accuracy:')
    lines.append(', '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
elif Exp == 3:
    lines.append('CoreSet Accuracy:')
    lines.append(', '.join(str(e) for e in CoreSetAccuracy_seed))

#write('results.txt', lines, CoreSet, alg2lol, '')

if SeedSet:
    if Exp == 1:
        corpusRandom = Random.setSeedSet()
        Random.printCurrentTrainingData()
        corpusConfidenceScores = ConfidenceScores.setSeedSet()
        ConfidenceScores.printCurrentTrainingData()
        Random.trainTARS(filename_model)
        ConfidenceScores.trainTARS(filename_model2)
        RandomAccuracy_seed.append(Random.evaluateModel())
        ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
    elif Exp == 2:
        corpusExpectedGradientLength = ExpectedGradientLength.setSeedSet()
        ExpectedGradientLength.trainTARS(filename_model)
        ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
    elif Exp == 3:
        corpusCoreSet = CoreSet.setSeedSet()
        CoreSet.trainTARS(filename_model)
        CoreSetAccuracy_seed.append(CoreSet.evaluateModel())

    lines.append('Finished Seed Training')
    if Exp == 1:
        lines.append('Random Accuracy:')
        lines.append(', '.join(str(e) for e in RandomAccuracy_seed))
        lines.append('Random Train Data:')
        for data in corpusRandom.train:
            lines.append(str(data))
        lines.append('ConfidenceScores Accuracy:')
        lines.append(', '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
        lines.append('ConfidenceScores Train Data:')
        for data in corpusConfidenceScores.train:
            lines.append(str(data))
    elif Exp == 2:
        lines.append('ExpectedGradientLength Accuracy:')
        lines.append(', '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
        lines.append('ExpectedGradientLength Train Data:')
        for data in corpusExpectedGradientLength.train:
            lines.append(str(data))
    elif Exp == 3:
        lines.append('CoreSet Accuracy:')
        lines.append(', '.join(str(e) for e in CoreSetAccuracy_seed))
        lines.append('CoreSet Train Data:')
        for data in corpusCoreSet.train:
            lines.append(str(data))
    write('results.txt', lines)#, ExpectedGradientLength, alg2lol, '')

for i in range(10):
    if Exp == 1:
        corpusRandom = Random.SelectData(TrainSetSize)
        Random.trainTARS(path = filename_model)
        RandomAccuracy_seed.append(Random.evaluateModel())
        corpusConfidenceScores = ConfidenceScores.SelectData(TrainSetSize)
        ConfidenceScores.trainTARS(path = filename_model2)
        ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
        print('Random Accuracy:')
        print(RandomAccuracy_seed)
        print('ConfidenceScores Accuracy:')
        print(ConfidenceScoresAccuracy_seed)
        lines.append(f'Ran {i}th active learning step:')
        lines.append('Random Accuracy:')
        lines.append(', '.join(str(e) for e in RandomAccuracy_seed))
        lines.append('Random Train Data:')
        for data in corpusRandom.train:
            lines.append(str(data))
        lines.append('ConfidenceScores Accuracy:')
        lines.append(', '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
        lines.append('ConfidenceScores Train Data:')
        for data in corpusConfidenceScores.train:
            lines.append(str(data))
    elif Exp == 2:
        corpusExpectedGradientLength = ExpectedGradientLength.SelectData(TrainSetSize,filename_model)
        ExpectedGradientLength.trainTARS(path = filename_model)
        ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
        print('ExpectedGradientLength Accuracy:')
        print(ExpectedGradientLengthAccuracy_seed)
        lines.append(f'Ran {i}th active learning step:')
        lines.append('ExpectedGradientLength Accuracy:')
        lines.append(', '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
        lines.append('ExpectedGradientLength Train Data:')
        for data in corpusExpectedGradientLength.train:
            lines.append(str(data))
    elif Exp == 3:
        corpusCoreSet = CoreSet.SelectData(TrainSetSize)
        CoreSet.trainTARS(path = filename_model)
        CoreSetAccuracy_seed.append(CoreSet.evaluateModel())
        print('CoreSet Accuracy:')
        print(CoreSetAccuracy_seed)
        lines.append(f'Ran {i}th active learning step:')
        lines.append('CoreSet Accuracy:')
        lines.append(', '.join(str(e) for e in CoreSetAccuracy_seed))
        lines.append('CoreSet Train Data:')
        for data in corpusCoreSet.train:
            lines.append(str(data))
    write('results.txt', lines)#, ExpectedGradientLength, alg2lol, '')
    TrainSetSize = 50
