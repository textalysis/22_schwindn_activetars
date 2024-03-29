import numpy as np
import flair.datasets
from Definitions import label_name_map_50, label_name_map_stackoverflow
# Using readlines()
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
file1 = open('/Users/niklasschwind/PycharmProjects/TARSxActiveLearning/ConfScoreDataStack.txt', 'r')
file2 = open('/Users/niklasschwind/PycharmProjects/TARSxActiveLearning/ExpGradDataStack.txt', 'r')
file3 = open('/Users/niklasschwind/PycharmProjects/TARSxActiveLearning/CoreSetDataStack.txt', 'r')
Lines = file1.readlines()
LinesExp = file2.readlines()
LinesCoreSet = file3.readlines()

count = 0
# Strips the newline character
#TREC = flair.datasets.TREC_50(label_name_map=label_name_map_50)
TREC: Corpus = ClassificationCorpus('/Users/niklasschwind/.flair/datasets/stackoverflow',
                                                 test_file='test.txt',
                                                 dev_file='dev.txt',
                                                 train_file='train.txt',
                                                 label_type='topic',
                                                 label_name_map=label_name_map_stackoverflow,
                                                 )
TRECTrainSentences = []
TRECTestSentences = []
RandomSentence = []
ConfScoresSentences = []
ExpGradSentences = []
CoreSetSentences = []
Random = False
ConfScores = False
for line in Lines:
    if len(line[:-1].split('→')) == 1:
        if line[:-1].split('→') == ['Random Train Data:']:
            Random = True
            ConfScores = False
        elif line[:-1].split('→') == ['ConfidenceScores Train Data:']:
            Random = False
            ConfScores = True
    elif len(line[:-1].split('→')) == 2 and Random:
        RandomSentence.append(line[:-1].split('→'))
    elif len(line[:-1].split('→')) == 2 and ConfScores:
        ConfScoresSentences.append(line[:-1].split('→'))
for line in LinesExp:
    if len(line[:-1].split('→')) == 2:
        ExpGradSentences.append(line[:-1].split('→'))

for line in LinesCoreSet:
    if len(line[:-1].split('→')) == 2:
        CoreSetSentences.append(line[:-1].split('→'))

for sentence in TREC.train:
    line = str(sentence)
    if len(line[:-1].split('→')) == 2:
        TRECTrainSentences.append(line.split('→'))
for sentence in TREC.test:
    line = str(sentence)
    if len(line[:-1].split('→')) == 2:
        TRECTestSentences.append(line.split('→'))

TRECTrainSentencesText = [content[0] for content in TRECTrainSentences]
TRECTrainSentencesLabels = [content[1] for content in TRECTrainSentences]
TRECTestSentencesText = [content[0] for content in TRECTestSentences]
TRECTestSentencesLabels = [content[1] for content in TRECTestSentences]

LabelProbTRECTrain = {element : TRECTrainSentencesLabels.count(element)/len(TRECTrainSentencesLabels)  for element in set(TRECTrainSentencesLabels)}
LabelProbTRECTest = {element : TRECTestSentencesLabels.count(element)/len(TRECTestSentencesLabels)  for element in set(TRECTrainSentencesLabels)}

MostUnlikelyTrain = dict(sorted(LabelProbTRECTrain.items(), key=lambda item: item[1]))
MostUnlikelyTest = dict(sorted(LabelProbTRECTest.items(), key=lambda item: item[1]))

LabelProbsListTrain = [LabelProbTRECTrain[element] for element in MostUnlikelyTrain.keys()]
LabelProbsListTest = [LabelProbTRECTest[element] for element in MostUnlikelyTrain.keys()]

TenMostUnlikelyTrain = {k: MostUnlikelyTrain[k] for k in list(MostUnlikelyTrain)[:10]}
TenMostUnlikelyTest = {k: MostUnlikelyTest[k] for k in list(MostUnlikelyTest)[:10]}
NotRepresentedInTest = {k: MostUnlikelyTest[k] for k in list(MostUnlikelyTest)[:8]}
#NotRepresentedInTest.pop(' question about creative material (1.0)') #since this is  the only one quiet likely in train Optional
print(NotRepresentedInTest)
print(MostUnlikelyTrain)
'''
from matplotlib import pyplot as plt
plt.style.use('dark_background')
plt.plot(  range(0,500,50),[0.2,0.4,0.55,0.65,0.73,0.79, 0.83, 0.85, 0.86, 0.87] ,label = 'Random')
plt.plot(  range(0,500,50),[0.2,0.5,0.65,0.72,0.80,0.82, 0.84, 0.86, 0.87, 0.87] ,label = 'Active Learning Algorithem')
#plt.plot(range(20), LabelProbsListTest, label = 'Label Likelyhoods Test')

plt.title('Model trained with Data selected randomly or by Active Learning Algorithem')
plt.xlabel('Size Train Set')
plt.ylabel('Accuracy Test Set')
#plt.ylim(0, 1)
#x = [0,2,4,6,8,10,12,14,16,18,20]#[0,5,10,15,20,25,30,35,40,45,50]
#ticks = [0,2,4,6,8,10,12,14,16,18,20]
#plt.xticks(x, ticks)
plt.legend()
plt.show()
'''
maxNumber = 0
maxItem = ''
for label, likelihood in LabelProbTRECTest.items():
    if likelihood > maxNumber:
        maxNumber = likelihood
        maxItem = label
print(maxItem)
maxNumber = 0
maxItem = ''
for label, likelihood in LabelProbTRECTrain.items():
    if likelihood > maxNumber:
        maxNumber = likelihood
        maxItem = label
print(maxItem)




RandomTrainSteps= []
ConfScoresTrainSteps= []
ExpGradTrainSteps = []
CoreSetTrainSteps = []
Down = 0
Up = 99
for i in range(11):
    RandomTrainSteps.append(RandomSentence[Down:Up])
    ConfScoresTrainSteps.append(ConfScoresSentences[Down:Up])
    ExpGradTrainSteps.append(ExpGradSentences[Down:Up])
    CoreSetTrainSteps.append(CoreSetSentences[Down:Up])
    Down = Up+1
    Up = Up+50




RandomSentencesText = [[content[0] for content in step] for step in RandomTrainSteps]
ConfScoresSentencesText = [[content[0] for content in step] for step in ConfScoresTrainSteps]
RandomSentencesLabels = [[content[1] for content in step] for step in RandomTrainSteps]
ConfScoresSentencesLabels = [[content[1] for content in step] for step in ConfScoresTrainSteps]
ExpGradSentencesText = [[content[0] for content in step] for step in ExpGradTrainSteps]
ExpGradSentencesLabels = [[content[1] for content in step] for step in ExpGradTrainSteps]
CoreSetSentencesText = [[content[0] for content in step] for step in CoreSetTrainSteps]
CoreSetSentencesLabels = [[content[1] for content in step] for step in CoreSetTrainSteps]


print(RandomSentencesLabels)
print(ConfScoresSentencesLabels)
print(ExpGradSentencesLabels)
print(CoreSetSentencesLabels)

NumberOfSameLabelsRandom = []
NumberOfSameLabelsConfScores = []
NumberOfSameLabelsExpGrad = []
NumberOfSameLabelsCoreSet = []
NumberOfMaxLabelMentionsRandom = []
NumberOfMaxLabelMentionsConfScores = []
NumberOfMaxLabelMentionsExpGrad = []
NumberOfMaxLabelMentionsCoreSet = []
RandomLabelEntropy = []
ConfScoresLabelEntropy = []
ExpGradLabelEntropy = []
CoreSetLabelEntropy = []
NumberQuestionAboutDefRandom = []
NumberQuestionAboutDefConfScores = []
NumberQuestionAboutDefExpGrad = []
NumberQuestionAboutDefCoreSet = []
NumberQuestionAboutIndRandom = []
NumberQuestionAboutIndConfScores = []
NumberQuestionAboutIndExpGrad = []
NumberQuestionAboutIndCoreSet = []
NumberQuestionNotRepTestRandom = []
NumberQuestionNotRepTestConfScores = []
NumberQuestionNotRepTestExpGrad = []
NumberQuestionNotRepTestCoreSet = []
NumberQuestionTenMostUnlikelyTrainRandom = []
NumberQuestionTenMostUnlikelyTrainConfScores = []
NumberQuestionTenMostUnlikelyTrainExpGrad = []
NumberQuestionTenMostUnlikelyTrainCoreSet = []
NumberQuestionAddedPropRandom = []
NumberQuestionAddedPropConfScores = []
NumberQuestionAddedPropExpGrad = []
NumberQuestionAddedPropCoreSet = []
NumberQuestionMultPropRandom = []
NumberQuestionMultPropConfScores = []
NumberQuestionMultPropExpGrad = []
NumberQuestionMultPropCoreSet = []
NumberQuestionAddedPropTestRandom = []
NumberQuestionAddedPropTestConfScores = []
NumberQuestionAddedPropTestExpGrad = []
NumberQuestionAddedPropTestCoreSet = []
AccuracyRandom = [0.452, 0.634, 0.704, 0.752, 0.718, 0.744, 0.696, 0.75, 0.776, 0.79, 0.79]
AccuracyConfScores = [0.676, 0.688, 0.48, 0.544, 0.576, 0.79, 0.764, 0.804, 0.802, 0.8, 0.778]
AccuracyExpGrad = [0.676, 0.664, 0.716, 0.504, 0.71, 0.744, 0.688, 0.56, 0.762, 0.796, 0.794]
AccuracyCoreSet = [0.71, 0.676, 0.732, 0.738, 0.572, 0.712, 0.754, 0.666, 0.718, 0.802, 0.746]
for list in RandomSentencesLabels:
    print(MostUnlikelyTrain)
    NumberOfSameLabelsRandom.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsRandom.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1*list.count(element)/len(list)*np.log(list.count(element)/len(list))
    RandomLabelEntropy.append(entropy)

    NumberQuestionAboutDefRandom.append(list.count(' question about the definition of something (1.0)'))
    NumberQuestionAboutIndRandom.append(list.count(' question about an individual (1.0)'))
    numberrep = 0
    for element in NotRepresentedInTest.keys():
        numberrep += list.count(element)
    NumberQuestionNotRepTestRandom.append(numberrep)
    numberrep = 0
    for element in TenMostUnlikelyTrain.keys():
        numberrep += list.count(element)
    NumberQuestionTenMostUnlikelyTrainRandom.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTrain[element]
    NumberQuestionAddedPropRandom.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep *= MostUnlikelyTrain[element]
    NumberQuestionMultPropRandom.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTest[element]
    NumberQuestionAddedPropTestRandom.append(numberrep)

for list in ConfScoresSentencesLabels:
    NumberOfSameLabelsConfScores.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsConfScores.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1 * list.count(element) / len(list) * np.log(list.count(element) / len(list))
    ConfScoresLabelEntropy.append(entropy)

    NumberQuestionAboutDefConfScores.append(list.count(' question about the definition of something (1.0)'))
    NumberQuestionAboutIndConfScores.append(list.count(' question about an individual (1.0)'))
    numberrep = 0
    for element in NotRepresentedInTest.keys():
        numberrep += list.count(element)
    NumberQuestionNotRepTestConfScores.append(numberrep)
    numberrep=0
    for element in TenMostUnlikelyTrain.keys():
        numberrep += list.count(element)
    NumberQuestionTenMostUnlikelyTrainConfScores.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTrain[element]
    NumberQuestionAddedPropConfScores.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep *= MostUnlikelyTrain[element]
    NumberQuestionMultPropConfScores.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTest[element]
    NumberQuestionAddedPropTestConfScores.append(numberrep)

for list in ExpGradSentencesLabels:
    NumberOfSameLabelsExpGrad.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsExpGrad.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1 * list.count(element) / len(list) * np.log(list.count(element) / len(list))
    ExpGradLabelEntropy.append(entropy)
    NumberQuestionAboutDefExpGrad.append(list.count(' question about the definition of something (1.0)'))
    NumberQuestionAboutIndExpGrad.append(list.count(' question about an individual (1.0)'))
    numberrep = 0

    for element in NotRepresentedInTest.keys():
        numberrep += list.count(element)
    NumberQuestionNotRepTestExpGrad.append(numberrep)
    numberrep = 0
    for element in TenMostUnlikelyTrain.keys():
        numberrep += list.count(element)
    NumberQuestionTenMostUnlikelyTrainExpGrad.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTrain[element]
    NumberQuestionAddedPropExpGrad.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep *= MostUnlikelyTrain[element]
    NumberQuestionMultPropExpGrad.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTest[element]
    NumberQuestionAddedPropTestExpGrad.append(numberrep)

for list in CoreSetSentencesLabels:
    NumberOfSameLabelsCoreSet.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsCoreSet.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1 * list.count(element) / len(list) * np.log(list.count(element) / len(list))
    CoreSetLabelEntropy.append(entropy)

    NumberQuestionAboutDefCoreSet.append(list.count(' question about the definition of something (1.0)'))
    NumberQuestionAboutIndCoreSet.append(list.count(' question about an individual (1.0)'))
    numberrep = 0
    for element in NotRepresentedInTest.keys():
        numberrep += list.count(element)
    NumberQuestionNotRepTestCoreSet.append(numberrep)
    numberrep = 0
    for element in TenMostUnlikelyTrain.keys():
        numberrep += list.count(element)
    NumberQuestionTenMostUnlikelyTrainCoreSet.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTrain[element]
    NumberQuestionAddedPropCoreSet.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep *= MostUnlikelyTrain[element]
    NumberQuestionMultPropCoreSet.append(numberrep)
    numberrep = 0
    for element in list:
        numberrep += MostUnlikelyTest[element]
    NumberQuestionAddedPropTestCoreSet.append(numberrep)

'''
from matplotlib import pyplot as plt
plt.plot(range(11), NumberOfMaxLabelMentionsRandom, label = 'Added Prob labels Random')
plt.plot(range(11), NumberOfMaxLabelMentionsConfScores, label = 'Added Prob labels ConfScore')
plt.plot(range(11), NumberOfMaxLabelMentionsExpGrad, label = 'Added Prob labels ExpGrad')
plt.plot(range(11), NumberOfMaxLabelMentionsCoreSet, label = 'Added Prob labels CoreSet')
plt.legend()
plt.show()
'''
#sentence = Sentence(NoLabelSentence)
#sentence.add_label(typename= self.LabelType, value=label)
Random_longer_acc = [0.228,  0.688, 0.706, 0.74, 0.726, 0.772, 0.77, 0.804, 0.772, 0.77, 0.782, 0.832, 0.806, 0.826, 0.834, 0.838, 0.828, 0.818, 0.842, 0.846, 0.836]
ConfScores_longer_acc = [0.228,  0.472, 0.698, 0.69, 0.668, 0.724, 0.742, 0.772, 0.784, 0.794, 0.842, 0.822, 0.832, 0.818, 0.836, 0.836, 0.842, 0.844, 0.876, 0.852, 0.836]
ExpGrad_longer_acc = [0.228, 0.366, 0.334, 0.456, 0.508, 0.512, 0.742, 0.75, 0.788, 0.794, 0.79, 0.81, 0.794, 0.796, 0.834, 0.84, 0.838, 0.836, 0.834, 0.842, 0.804]

from matplotlib import pyplot as plt
plt.style.use('dark_background')
plt.plot(range(10), NumberOfMaxLabelMentionsRandom[1:], label = 'Random')
plt.plot(range(10), NumberOfMaxLabelMentionsConfScores[1:], label = 'Confidence Scores')
plt.plot(range(10), NumberOfMaxLabelMentionsExpGrad[1:], label = 'Expected Gradient Length')
plt.plot(range(10), NumberOfMaxLabelMentionsCoreSet[1:], label = 'Core Set')
#plt.plot(range(11), NumberQuestionTenMostUnlikelyTrainCoreSet, label = '10 most unlikely labels mentions CoreSet')
plt.title('Number of Representations of most represented Label - Stack Overflow')
plt.xlabel('Run')
plt.ylabel('Number of Representations in Selected Dataset')

x = [0,1,2,3,4,5,6,7,8,9]
ticks = [1,2,3,4,5,6,7,8,9,10]
plt.xticks(x, ticks)
plt.legend()
plt.show()
