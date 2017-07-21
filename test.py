from NN.test import draw
from argparse import Namespace


args = {
    'batch_size': 3,
    'lr': 1e-5,
    'beta1': 0.9,
    'max_epochs': 2,
    'save_freq': 1,
    'summary_freq': 0,
    'progress_freq': 1,
    'validation_freq': 1,
    'checkpoint': ''
}

a = Namespace(**args)
draw('./data/generated/010', './data/generated/kmeans', a,
     ['./data/trained_model/000/model-1', './data/trained_model/001/model-1'])
