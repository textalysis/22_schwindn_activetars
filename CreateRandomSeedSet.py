import flair.datasets
from flair.models import TARSClassifier

import Random as Rand
TARS_Random = TARSClassifier.load('tars-base')
TREC_Random = flair.datasets.TREC_50().downsample(1)
Random = Rand.Random(corpus = TREC_Random, TARS = TARS_Random)
path = '/vol/fob-vol7/mi19/schwindn/.flair/datasets/trec_50/'
with open(path+'train.txt', 'w') as f:
    for data in TREC_Random.train:
        f.write(f'__label__{data.labels[0].value} {data.to_plain_string()}\n')
with open(path+'dev.txt', 'w') as f:
    for data in TREC_Random.dev:
        f.write(f'__label__{data.labels[0].value} {data.to_plain_string()}\n')
with open(path+'test.txt', 'w') as f:
    for data in TREC_Random.test:
        f.write(f'__label__{data.labels[0].value} {data.to_plain_string()}\n')
