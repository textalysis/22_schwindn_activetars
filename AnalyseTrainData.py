import numpy as np
# Using readlines()
file1 = open('/Users/niklasschwind/PycharmProjects/TARSxActiveLearning/ConfScoreData.txt', 'r')
file2 = open('/Users/niklasschwind/PycharmProjects/TARSxActiveLearning/ExpGradData.txt', 'r')
Lines = file1.readlines()
LinesExp = file2.readlines()

count = 0
# Strips the newline character
RandomSentence = []
ConfScoresSentences = []
ExpGradSentences = []
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


RandomTrainSteps= []
ConfScoresTrainSteps= []
ExpGradTrainSteps = []
Down = 0
Up = 99
for i in range(11):
    RandomTrainSteps.append(RandomSentence[Down:Up])
    ConfScoresTrainSteps.append(ConfScoresSentences[Down:Up])
    ExpGradTrainSteps.append(ExpGradSentences[Down:Up])
    Down = Up+1
    Up = Up+50




RandomSentencesText = [[content[0] for content in step] for step in RandomTrainSteps]
ConfScoresSentencesText = [[content[0] for content in step] for step in ConfScoresTrainSteps]
RandomSentencesLabels = [[content[1] for content in step] for step in RandomTrainSteps]
ConfScoresSentencesLabels = [[content[1] for content in step] for step in ConfScoresTrainSteps]
ExpGradSentencesText = [[content[0] for content in step] for step in ExpGradTrainSteps]
ExpGradSentencesLabels = [[content[1] for content in step] for step in ExpGradTrainSteps]


print(RandomSentencesLabels)
print(ConfScoresSentencesLabels)
print(ExpGradSentencesLabels)

NumberOfSameLabelsRandom = []
NumberOfSameLabelsConfScores = []
NumberOfSameLabelsExpGrad = []
NumberOfMaxLabelMentionsRandom = []
NumberOfMaxLabelMentionsConfScores = []
NumberOfMaxLabelMentionsExpGrad = []
RandomLabelEntropy = []
ConfScoresLabelEntropy = []
ExpGradLabelEntropy = []

for list in RandomSentencesLabels:
    NumberOfSameLabelsRandom.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsRandom.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1*list.count(element)/len(set(list))*np.log(list.count(element)/len(set(list)))
    RandomLabelEntropy.append(entropy)
for list in ConfScoresSentencesLabels:
    NumberOfSameLabelsConfScores.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsConfScores.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1 * list.count(element) / len(set(list)) * np.log(list.count(element) / len(set(list)))
    ConfScoresLabelEntropy.append(entropy)
for list in ExpGradSentencesLabels:
    NumberOfSameLabelsExpGrad.append(len(set(list)))
    max = 0
    for element in set(list):
        if list.count(element) > max:
            max = list.count(element)
    NumberOfMaxLabelMentionsExpGrad.append(max)
    entropy = 0
    for element in set(list):
        entropy += -1 * list.count(element) / len(set(list)) * np.log(list.count(element) / len(set(list)))
    ExpGradLabelEntropy.append(entropy)



from matplotlib import pyplot as plt
plt.plot(range(11), NumberOfSameLabelsRandom, label = '# Labels to be found in Random training data')
plt.plot(range(11), NumberOfSameLabelsConfScores, label = '# Labels to be found in ConfScoreTraining data')
plt.plot(range(11), NumberOfSameLabelsExpGrad, label = '# Labels to be found in ExpGradTraining data')

plt.legend()
plt.show()

#sentence = Sentence(NoLabelSentence)
#sentence.add_label(typename= self.LabelType, value=label)