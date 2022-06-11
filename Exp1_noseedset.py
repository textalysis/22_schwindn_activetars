import flair, torch
import flair.datasets
from flair.models import TARSClassifier
from Definitions import label_name_map_50
import Random as Rand
import ConfidenceScores as Conf
import ExpectedGradientLength as Expe
import CoreSet as Core


def write(name, contents):
    with open('results_ConfScor_withoutseedset.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(contents))

flair.device = torch.device('cuda:2')
#Run experiment with seedset
lines = []
#Initialize Models
TARS_Random = TARSClassifier.load('tars-base')
TARS_ConfidenceScores = TARSClassifier.load('tars-base')
#TARS_ExpectedGradientLength = TARSClassifier.load('tars-base')
#TARS_CoreSet = TARSClassifier.load('tars-base')

#Initialize Corpora
sample_value = 1
TREC_Random = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_ConfidenceScores = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
#TREC_ExpectedGradientLength = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
#TREC_CoreSet = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)

#Initialize ActiveLearners
Random = Rand.Random(corpus = TREC_Random, TARS = TARS_Random)
ConfidenceScores = Conf.ConfidenceScores(corpus = TREC_ConfidenceScores, TARS = TARS_ConfidenceScores)
#ExpectedGradientLength = Expe.ExpectedGradientLength(corpus = TREC_ExpectedGradientLength, TARS = TARS_ExpectedGradientLength)
#CoreSet = Core.CoreSet(corpus = TREC_CoreSet, TARS = TARS_CoreSet)

TrainSetSize = 50
BaseAccuracy = Random.evaluateModel()
RandomAccuracy_seed = [BaseAccuracy]
ConfidenceScoresAccuracy_seed = [BaseAccuracy]
ExpectedGradientLengthAccuracy_seed = [BaseAccuracy]
CoreSetAccuracy_seed = [BaseAccuracy]

lines.append('Initialized Models:')
lines.append('Random Accuracy:')
lines.append(' '.join(str(e) for e in RandomAccuracy_seed))
lines.append('ConfidenceScores Accuracy:')
lines.append(' '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
#lines.append('ExpectedGradientLength Accuracy:')
#lines.append(' '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
#lines.append('CoreSet Accuracy:')
#lines.append(' '.join(str(e) for e in CoreSetAccuracy_seed))
write('results.txt', lines)


#lines.append('ExpectedGradientLength Accuracy:')
#lines.append(' '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
#lines.append('CoreSet Accuracy:')
#lines.append(' '.join(str(e) for e in CoreSetAccuracy_seed))


#Random.SelectData(TrainSetSize)
#Random.trainTARS(path = 'resources/taggers/Random')
#RandomAccuracy_seed.append(Random.evaluateModel())
#ConfidenceScores.SelectRandomData(TrainSetSize)
#ConfidenceScores.trainTARS(path = 'resources/taggers/ConfidenceScores')
#ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
#ExpectedGradientLength.SelectRandomData(TrainSetSize)
#ExpectedGradientLength.trainTARS(path = 'resources/taggers/ExpectedGradientLength')
#ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
#CoreSet.SelectRandomData(TrainSetSize)
#CoreSet.trainTARS(path = 'resources/taggers/CoreSet')
#CoreSetAccuracy_seed.append(CoreSet.evaluateModel())
#TrainSetSize = 50
#lines.append('Ran seed set training:')
#lines.append('Random Accuracy:')
#lines.append(' '.join(str(e) for e in RandomAccuracy_seed))
#lines.append('ConfidenceScores Accuracy:')
#lines.append(' '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
#lines.append('ExpectedGradientLength Accuracy:')
#lines.append(' '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
#lines.append('CoreSet Accuracy:')
#lines.append(' '.join(str(e) for e in CoreSetAccuracy_seed))
#write('results.txt', lines)
for i in range(10):
    Random.SelectData(TrainSetSize)
    Random.trainTARS(path = 'resources/taggers/Random3')
    RandomAccuracy_seed.append(Random.evaluateModel())
    ConfidenceScores.SelectData(TrainSetSize)
    ConfidenceScores.trainTARS(path = 'resources/taggers/ConfidenceScores3')
    ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
   # ExpectedGradientLength.SelectData(TrainSetSize)
    #ExpectedGradientLength.trainTARS(path = 'resources/taggers/ExpectedGradientLength')
    #ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
    #CoreSet.SelectData(TrainSetSize)
    #CoreSet.trainTARS(path = 'resources/taggers/CoreSet')
    #CoreSetAccuracy_seed.append(CoreSet.evaluateModel())
    TrainSetSize = 50

    print('Random Accuracy:')
    print(RandomAccuracy_seed)
    print('ConfidenceScores Accuracy:')
    print(ConfidenceScoresAccuracy_seed)
    #print('ExpectedGradientLength Accuracy:')
    #print(ExpectedGradientLengthAccuracy_seed)
    #print('CoreSet Accuracy:')
    #print(CoreSetAccuracy_seed)
    lines.append(f'Ran {i}th active learning step:')
    lines.append('Random Accuracy:')
    lines.append(' '.join(str(e) for e in RandomAccuracy_seed))
    lines.append('ConfidenceScores Accuracy:')
    lines.append(' '.join(str(e) for e in ConfidenceScoresAccuracy_seed))
    #lines.append('ExpectedGradientLength Accuracy:')
    #lines.append(' '.join(str(e) for e in ExpectedGradientLengthAccuracy_seed))
    #lines.append('CoreSet Accuracy:')
    #lines.append(' '.join(str(e) for e in CoreSetAccuracy_seed))
    write('results.txt', lines)