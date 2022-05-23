import flair.datasets
from flair.models import TARSClassifier
from Definitions import label_name_map_50
import Random as Rand
import ConfidenceScores as Conf
import ExpectedGradientLength as Expe
import CoreSet as Core


#Run experiment with seedset

#Initialize Models
TARS_Random = TARSClassifier.load('tars-base')
TARS_ConfidenceScores = TARSClassifier.load('tars-base')
TARS_ExpectedGradientLength = TARSClassifier.load('tars-base')
TARS_CoreSet = TARSClassifier.load('tars-base')

#Initialize Corpora
sample_value = 1
TREC_Random = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_ConfidenceScores = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_ExpectedGradientLength = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_CoreSet = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)

#Initialize ActiveLearners
Random = Rand.Random(corpus = TREC_Random, TARS = TARS_Random)
ConfidenceScores = Conf.ConfidenceScores(corpus = TREC_ConfidenceScores, TARS = TARS_ConfidenceScores)
ExpectedGradientLength = Expe.ExpectedGradientLength(corpus = TREC_ExpectedGradientLength, TARS = TARS_ExpectedGradientLength)
CoreSet = Core.CoreSet(corpus = TREC_CoreSet, TARS = TARS_CoreSet)

TrainSetSize = 100
BaseAccuracy = Random.evaluateModel()
RandomAccuracy_seed = [BaseAccuracy]
ConfidenceScoresAccuracy_seed = [BaseAccuracy]
ExpectedGradientLengthAccuracy_seed = [BaseAccuracy]
CoreSetAccuracy_seed = [BaseAccuracy]
print(BaseAccuracy)
print(Random.CorpusLabels)



Random.SelectData(TrainSetSize)
Random.trainTARS(path = 'resources/taggers/Random')
RandomAccuracy_seed.append(Random.evaluateModel())
ConfidenceScores.SelectRandomData(TrainSetSize)
ConfidenceScores.trainTARS(path = 'resources/taggers/ConfidenceScores')
ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
ExpectedGradientLength.SelectRandomData(TrainSetSize)
ExpectedGradientLength.trainTARS(path = 'resources/taggers/ExpectedGradientLength')
ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
CoreSet.SelectRandomData(TrainSetSize)
CoreSet.trainTARS(path = 'resources/taggers/CoreSet')
CoreSetAccuracy_seed.append(CoreSet.evaluateModel())
TrainSetSize = 100
print('Random Accuracy:')
print(RandomAccuracy_seed)
print('ConfidenceScores Accuracy:')
print(ConfidenceScoresAccuracy_seed)
print('ExpectedGradientLength Accuracy:')
print(ExpectedGradientLengthAccuracy_seed)
print('CoreSet Accuracy:')
print(CoreSetAccuracy_seed)
for i in range(5):
    Random.SelectData(TrainSetSize)
    Random.trainTARS(path = 'resources/taggers/Random')
    RandomAccuracy_seed.append(Random.evaluateModel())
    ConfidenceScores.SelectData(TrainSetSize)
    ConfidenceScores.trainTARS(path = 'resources/taggers/ConfidenceScores')
    ConfidenceScoresAccuracy_seed.append(ConfidenceScores.evaluateModel())
    ExpectedGradientLength.SelectData(TrainSetSize)
    ExpectedGradientLength.trainTARS(path = 'resources/taggers/ExpectedGradientLength')
    ExpectedGradientLengthAccuracy_seed.append(ExpectedGradientLength.evaluateModel())
    CoreSet.SelectData(TrainSetSize)
    CoreSet.trainTARS(path = 'resources/taggers/CoreSet')
    CoreSetAccuracy_seed.append(CoreSet.evaluateModel())
    TrainSetSize = 50
    print('Random Accuracy:')
    print(RandomAccuracy_seed)
    print('ConfidenceScores Accuracy:')
    print(ConfidenceScoresAccuracy_seed)
    print('ExpectedGradientLength Accuracy:')
    print(ExpectedGradientLengthAccuracy_seed)
    print('CoreSet Accuracy:')
    print(CoreSetAccuracy_seed)


#Run experiments without seedset

#Initialize Models
TARS_Random = TARSClassifier.load('tars-base')
TARS_ConfidenceScores = TARSClassifier.load('tars-base')
TARS_ExpectedGradientLength = TARSClassifier.load('tars-base')
TARS_CoreSet = TARSClassifier.load('tars-base')

#Initialize Corpora
sample_value = 1
TREC_Random = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_ConfidenceScores = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_ExpectedGradientLength = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)
TREC_CoreSet = flair.datasets.TREC_50(label_name_map=label_name_map_50).downsample(sample_value)

#Initialize ActiveLearners
Random = Rand.Random(corpus = TREC_Random, TARS = TARS_Random)
ConfidenceScores = Conf.ConfidenceScores(corpus = TREC_ConfidenceScores, TARS = TARS_ConfidenceScores)
ExpectedGradientLength = Expe.ExpectedGradientLength(corpus = TREC_ExpectedGradientLength, TARS = TARS_ExpectedGradientLength)
CoreSet = Core.CoreSet(corpus = TREC_CoreSet, TARS = TARS_CoreSet)

BaseAccuracy = Random.evaluateModel()
RandomAccuracy_noseed = [BaseAccuracy]
ConfidenceScoresAccuracy_noseed = [BaseAccuracy]
ExpectedGradientLengthAccuracy_noseed = [BaseAccuracy]
CoreSetAccuracy_noseed = [BaseAccuracy]

for i in range(6):
    Random.SelectData(TrainSetSize)
    Random.trainTARS(path = 'resources/taggers/Random')
    RandomAccuracy_noseed.append(Random.evaluateModel())
    ConfidenceScores.SelectData(TrainSetSize)
    ConfidenceScores.trainTARS(path = 'resources/taggers/ConfidenceScores')
    ConfidenceScoresAccuracy_noseed.append(ConfidenceScores.evaluateModel())
    ExpectedGradientLength.SelectData(TrainSetSize)
    ExpectedGradientLength.trainTARS(path = 'resources/taggers/ExpectedGradientLength')
    ExpectedGradientLengthAccuracy_noseed.append(ExpectedGradientLength.evaluateModel())
    CoreSet.SelectData(TrainSetSize)
    CoreSet.trainTARS(path = 'resources/taggers/CoreSet')
    CoreSetAccuracy_noseed.append(CoreSet.evaluateModel())
    TrainSetSize = 5
    print('Random Accuracy:')
    print(RandomAccuracy_noseed)
    print('ConfidenceScores Accuracy:')
    print(ConfidenceScoresAccuracy_noseed)
    print('ExpectedGradientLength Accuracy:')
    print(ExpectedGradientLengthAccuracy_noseed)
    print('CoreSet Accuracy:')
    print(CoreSetAccuracy_noseed)



