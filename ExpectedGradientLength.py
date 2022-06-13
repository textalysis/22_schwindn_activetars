from ActiveLearner import ActiveLearner

import flair.datasets
from flair.data import Sentence
from flair.models import TARSClassifier
from flair.trainers import ModelTrainer

#Chooses NumberOfElements Elements according to Expected Gradient Length algorithm from unused trainingdata in basecorpus
class ExpectedGradientLength(ActiveLearner):
    def SelectData(self, NumberOfElements: int):
        ExpectedGradientLenghtForSentence = {}
        DataToIndex = {}
        IndexAndGradientTupleList = []
        self.TARS.add_and_switch_to_new_task(task_name="question classification",
                                label_dictionary=self.LabelDict,
                                label_type=self.LabelType,
                                )
        Index = 0
        for data in self.basecorpus.train:
            DataToIndex[data.to_plain_string()] = Index
            Index += 1

        batch_loader = flair.datasets.DataLoader(
                self.basecorpus.train,
                batch_size=1,
                shuffle=False,
                num_workers=None,
                sampler=None,
            )
        DummyModel = TARSClassifier.load('tars-base')
        DummyModel.add_and_switch_to_new_task(task_name=self.task,
                                        label_dictionary=self.LabelDict,
                                        label_type=self.LabelType,
                                        )
        if self.currentTrainCorpus != []:
            trainer = ModelTrainer(DummyModel, self.currentTrainCorpus)
            trainer.train(base_path=f'resources/taggers/DummyModel',  # path to store the model artifacts
                          learning_rate=self.learning_rate,  # use very small learning rate
                          mini_batch_size=self.mini_batch_size,
                          mini_batch_chunk_size=self.mini_batch_chunk_size,  # optionally set this if transformer is too much for your machine
                          max_epochs=self.max_epochs,  # terminate after 20 epochs
                          shuffle=self.shuffle,
                          )

        for i, TrueLabelSentence in enumerate(batch_loader):
            Gradients_of_each_Lables = []
            NoLabelSentence = TrueLabelSentence[0].to_plain_string()
            for label in self.CorpusLabels:
                sentence = Sentence(NoLabelSentence)
                sentence.add_label(typename= self.LabelType, value=label)
                batch = [sentence]
                total_norm = 0
                loss = DummyModel.forward_loss(batch)
                loss[0].backward()
                for p in DummyModel.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                        p.grad.data.zero_()
                total_norm = total_norm ** 0.5
                Gradients_of_each_Lables.append(total_norm)
            print(f'Calculating Gradient for datapoint {i}')
            ExpectedGradientLenghtForSentence[batch[0].to_plain_string()] = min(Gradients_of_each_Lables)
        for key in ExpectedGradientLenghtForSentence.keys():
            IndexAndGradientTupleList.append((DataToIndex[key],
                                              ExpectedGradientLenghtForSentence[key]))
        IndexAndGradientTupleList_sorted = sorted(IndexAndGradientTupleList,
                                                  key=lambda tup: -tup[1])
        IndexAndGradientTupleList_sorted = [i for i in IndexAndGradientTupleList_sorted
                                            if i[0] not in self.UsedIndices]
        SelectedIndices = [i[0] for i in IndexAndGradientTupleList_sorted[:NumberOfElements]]
        self.UsedIndices.extend(SelectedIndices)
        return self.downsampleCorpus(IndicesToKeep = self.UsedIndices)




