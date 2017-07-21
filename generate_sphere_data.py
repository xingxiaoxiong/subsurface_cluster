import numpy as np
import os
from sklearn.cluster import KMeans


from data_generator.generate_training_data_sphere import generate_data


def generate_single_sphere(sources, destination):
    if not os.path.isdir(destination):
        os.makedirs(destination)

    count = 0
    for source in sources:
        count = generate_data(source, count, destination)


if __name__ == '__main__':
    generate_single_sphere(['./data/bin/010'], './data/generated/010')
