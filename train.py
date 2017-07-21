
from NN.train import train
import os
from argparse import Namespace


sources = ['./data/clustered/000', './data/clustered/001']

args = {
    'batch_size': 3,
    'lr': 1e-5,
    'beta1': 0.9,
    'max_epochs': 2,
    'save_freq': 1,
    'summary_freq': 0,
    'progress_freq': 1,
    'validation_freq': 1,
    'checkpoint': None
}

for source in sources:
    cluster_name = source.split('/')[-1]
    args['source_dir'] = source
    args['output_dir'] = os.path.join('./data/trained_model', cluster_name)
    a = Namespace(**args)
    train(a)

