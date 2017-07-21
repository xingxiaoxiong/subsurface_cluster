
import tensorflow as tf
import numpy as np
import os
import pickle

from .model import CNN
from collections import defaultdict


def calc_labels(data_dir, kmeans_path):
    front_position = np.load(os.path.join(data_dir, 'front_position.npy'))
    back_position = np.load(os.path.join(data_dir, 'back_position.npy'))
    object_mask = np.load(os.path.join(data_dir, 'object_mask.npy'))
    height, width, _ = front_position.shape

    kmeans_file = open(kmeans_path, 'rb')
    kmeans = pickle.load(kmeans_file)

    pixel_label = np.zeros((height, width), dtype=np.int32)
    for h in range(height):
        for w in range(width):
            position = front_position[h, w]
            if position[0] == 0.0 and position[1] == 0.0 and position[2] == 0.0:
                pixel_label[h, w] = -1
            else:
                front_relative_position = front_position - position
                back_relative_position = back_position - position
                front_relative_position *= object_mask
                back_relative_position *= object_mask

                front_relative_distance = np.sqrt(
                    front_relative_position[:, :, 0] * front_relative_position[:, :, 0] + front_relative_position[:,
                                                                                          :,
                                                                                          1] * front_relative_position[
                                                                                               :, :,
                                                                                               1] + front_relative_position[
                                                                                                    :, :,
                                                                                                    2] * front_relative_position[
                                                                                                         :, :, 2])
                back_relative_distance = np.sqrt(
                    back_relative_position[:, :, 0] * back_relative_position[:, :, 0] + back_relative_position[:, :,
                                                                                        1] * back_relative_position[
                                                                                             :, :,
                                                                                             1] + back_relative_position[
                                                                                                  :,
                                                                                                  :,
                                                                                                  2] * back_relative_position[
                                                                                                       :, :, 2])

                front_relative_distance = front_relative_distance[..., None]
                back_relative_distance = back_relative_distance[..., None]
                X = np.concatenate((front_relative_distance, back_relative_distance), axis=2).reshape((1, -1))
                label = kmeans.predict(X)
                pixel_label[h, w] = label[0]

    # from PIL import Image
    # label_image = (pixel_label + 1.0) * 125
    # label_image = label_image.astype('uint32')
    # img = Image.fromarray(label_image)
    # img.show()
    return pixel_label


def draw(data_dir, kmeans_path, a, checkpoints):
    front_position = np.load(os.path.join(data_dir, 'front_position.npy'))
    back_position = np.load(os.path.join(data_dir, 'back_position.npy'))
    front_lit = np.load(os.path.join(data_dir, 'front_irradiance.npy'))
    back_lit = np.load(os.path.join(data_dir, 'back_irradiance.npy'))
    object_mask = np.load(os.path.join(data_dir, 'object_mask.npy'))
    height, width, _ = front_position.shape
    depth = 8

    pixel_label = calc_labels(data_dir, kmeans_path)
    clusters = defaultdict(list)
    for h in range(height):
        for w in range(width):
            if pixel_label[h, w] != -1:
                clusters[pixel_label[h, w]].append([h, w])

    model = CNN(height, width, depth, a)
    model.build_graph(False, False)
    saver = tf.train.Saver()

    batch_size = 20
    image = np.zeros((height, width, 3))

    with tf.Session() as sess:
        for cluster_index, positions in clusters.items():

            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoints[cluster_index])

            for start_index in range(0, len(positions), batch_size):
                batch = positions[start_index: start_index + batch_size]

                X = np.zeros(shape=(len(batch), height, width, depth), dtype=np.float32)
                for i, pos in enumerate(batch):
                    position = front_position[pos[0], pos[1]]
                    front_relative_position = front_position - position
                    back_relative_position = back_position - position
                    front_relative_position *= object_mask
                    back_relative_position *= object_mask

                    front_relative_distance = np.sqrt(
                        front_relative_position[:, :, 0] * front_relative_position[:, :, 0] + front_relative_position[:, :,
                                                                                              1] * front_relative_position[
                                                                                                   :, :,
                                                                                                   1] + front_relative_position[
                                                                                                        :, :,
                                                                                                        2] * front_relative_position[
                                                                                                             :, :, 2])
                    back_relative_distance = np.sqrt(
                        back_relative_position[:, :, 0] * back_relative_position[:, :, 0] + back_relative_position[:, :,
                                                                                            1] * back_relative_position[:,
                                                                                                 :,
                                                                                                 1] + back_relative_position[
                                                                                                      :,
                                                                                                      :,
                                                                                                      2] * back_relative_position[
                                                                                                           :, :, 2])
                    front_relative_distance = front_relative_distance[..., None]
                    back_relative_distance = back_relative_distance[..., None]

                    X[i] = np.concatenate((front_relative_distance, back_relative_distance, front_lit, back_lit), axis=2)
                colors = sess.run(model.color, {model.input: X})
                for i, color in enumerate(colors):
                    pos = batch[i]
                    image[pos[0], pos[1]] = color

    np.save('output', image)

if __name__ == '__main__':
    pass
