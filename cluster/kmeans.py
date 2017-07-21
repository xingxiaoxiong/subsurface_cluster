import os
import re
from constants import *
import shutil
from sklearn.cluster import KMeans
import numpy as np
import pickle


def run_cluster(source, n_clusters):
    filenames = os.listdir(source)
    filenames = [filename for filename in filenames if re.match(npy_regx, filename)]
    filenames = sorted(filenames)
    data = []
    for filename in filenames:
        path = os.path.join(source, filename)
        datum = np.load(path)
        data.append(datum[0][:, :, 0:2].reshape(-1))
    data = np.asarray(data)
    kmeans = KMeans(n_clusters).fit(data)
    with open('./data/generated/kmeans', 'wb') as file:
        pickle.dump(kmeans, file)

    for i in range(n_clusters):
        dir = os.path.join(cluster_root, cluster_dir_format % i)
        if not os.path.isdir(dir):
            os.makedirs(dir)

    for i, label in enumerate(kmeans.labels_):
        shutil.copyfile(os.path.join(source, filenames[i]), os.path.join(cluster_root, cluster_dir_format % label, filenames[i]))


def read_kmeans(source):
    with open(source, 'rb') as file:
        kmeans = pickle.load(file)
        print(kmeans.labels_)


# def divide_data(data_dir, kmeans_path):
#     filenames = os.listdir(data_dir)
#     filenames = [filename for filename in filenames if re.match(npy_regx, filename)]
#     filenames = sorted(filenames)
#
#     with open(kmeans_path, 'rb') as file:
#         kmeans = pickle.load(file)
#         kmeans.labels_


