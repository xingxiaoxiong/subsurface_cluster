import scipy.ndimage
from constants import npy_format
import numpy as np
import os
import random
from PIL import Image

from .bin_viewer import read_bin

random.seed(1)


def generate_data(data_dir, count, save_dir):

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    back_position_path = os.path.join(data_dir, 'sphere_up_back_position.bin')
    back_position_data = read_bin(back_position_path)
    back_position_data = scipy.ndimage.zoom(input=back_position_data, zoom=(0.5, 0.5, 1), order=1)

    front_position_path = os.path.join(data_dir, 'sphere_up_front_position.bin')
    front_position_data = read_bin(front_position_path)
    front_position_data = scipy.ndimage.zoom(input=front_position_data, zoom=(0.5, 0.5, 1), order=1)

    back_irradiance_path = os.path.join(data_dir, 'sphere_up_back_irradiance.bin')
    back_irradiance_data = read_bin(back_irradiance_path)
    back_irradiance_data = scipy.ndimage.zoom(input=back_irradiance_data, zoom=(0.5, 0.5, 1), order=1)

    front_irradiance_path = os.path.join(data_dir, 'sphere_up_front_irradiance.bin')
    front_irradiance_data = read_bin(front_irradiance_path)
    front_irradiance_data = scipy.ndimage.zoom(input=front_irradiance_data, zoom=(0.5, 0.5, 1), order=1)

    output_path = os.path.join(data_dir, 'sphere_up_output.bin')
    output_data = read_bin(output_path)
    output_data = scipy.ndimage.zoom(input=output_data, zoom=(0.5, 0.5, 1), order=1)

    height, width, _ = output_data.shape

    object_mask = np.zeros((height, width, 3)).astype('int')
    for h in range(height):
        for w in range(width):
            target_point_position = front_position_data[h, w]
            if target_point_position[0] != 0.0 or target_point_position[1] != 0.0 or target_point_position[2] != 0.0:
                object_mask[h, w, :] = 1

    # img = Image.fromarray(object_mask.astype('uint8') * 255)
    # img.show()

    step_size = 16
    anchors_h = range(0, height, step_size)
    anchors_w = range(0, width, step_size)
    sample_number = 1

    # generate sample_mask
    sample_mask = np.zeros((height, width)).astype('uint8')
    for h in anchors_h:
        for w in anchors_w:
            for _ in range(sample_number):
                dh = random.randint(0, step_size - 1)
                dw = random.randint(0, step_size - 1)

                sample_h = h + dh
                sample_w = w + dw

                target_point_position = front_position_data[sample_h, sample_w]
                if target_point_position[0] != 0.0 or target_point_position[1] != 0.0 or target_point_position[2] != 0.0:
                    sample_mask[sample_h, sample_w] = 255

    for h in range(height):
        for w in range(width):
            if sample_mask[h, w] == 255:
                target_point_position = front_position_data[h, w]
                front_relative_position = front_position_data - target_point_position
                back_relative_position = back_position_data - target_point_position
                front_relative_position *= object_mask
                back_relative_position *= object_mask

                front_relative_distance = np.sqrt(front_relative_position[:, :, 0] * front_relative_position[:, :, 0] + front_relative_position[:, :, 1] * front_relative_position[:, :, 1] + front_relative_position[:, :, 2] * front_relative_position[:, :, 2])
                back_relative_distance = np.sqrt(back_relative_position[:, :, 0] * back_relative_position[:, :, 0] + back_relative_position[:, :, 1] * back_relative_position[:, :, 1] + back_relative_position[:, :, 2] * back_relative_position[:, :, 2])
                front_relative_distance = front_relative_distance[..., None]
                back_relative_distance = back_relative_distance[..., None]

                export_X = np.concatenate((front_relative_distance, back_relative_distance, front_irradiance_data, back_irradiance_data), axis=2)
                np.save(os.path.join(save_dir, npy_format % count), np.array((export_X, output_data[h, w])))
                count += 1

    np.save(os.path.join(save_dir, 'back_irradiance.npy'), back_irradiance_data)
    np.save(os.path.join(save_dir, 'front_irradiance.npy'), front_irradiance_data)
    np.save(os.path.join(save_dir, 'front_position.npy'), front_position_data)
    np.save(os.path.join(save_dir, 'back_position.npy'), back_position_data)
    np.save(os.path.join(save_dir, 'output.npy'), output_data)
    np.save(os.path.join(save_dir, 'object_mask.npy'), object_mask)

    img = Image.fromarray(sample_mask)
    img.save(os.path.join(save_dir, 'sample_mask.png'))
    return count



def check_generated_data():
    file_path = './data/00000.npy'
    data = np.load(file_path)
    print(data.shape)

def vis_position(path):
    data = np.load(path)
    print(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    print(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    print(np.min(data[:, :, 2]), np.max(data[:, :, 2]))
    data = (data + 1) * 0.5 * 255
    data = data.astype('uint8')
    img = Image.fromarray(data)
    img.show()
    img.save('x.jpg')
    print(np.min(data[:, :, 0]), np.max(data[:, :, 0]))
    print(np.min(data[:, :, 1]), np.max(data[:, :, 1]))
    print(np.min(data[:, :, 2]), np.max(data[:, :, 2]))

def vis_relative_position(path):
    slice_index = 0
    data = np.load(path)[:, :, slice_index]
    min_val = np.min(data)
    max_val = np.max(data)
    data -= min_val
    data /= (max_val - min_val)
    data *= 255
    data = data.astype('uint8')
    img = Image.fromarray(data)
    img.show()


def start_generating(base_dir):
    dirnames = set(os.listdir(base_dir))
    # training_dirs = {'010', '008', '006', '004', '002', '000'}
    training_dirs = dirnames
    validation_dirs = dirnames - training_dirs
    count = 0
    y = []
    for dir_name in training_dirs:
        data_path = os.path.join(base_dir, dir_name)
        count, y = generate_data(data_path, count, y)
    y = np.array(y)
    np.save(os.path.join('./data', 'y.npy'), y)


if __name__ == '__main__':
    # check_generated_data()
    start_generating('../data/blend2')
    # vis_position('./data/back_position.npy')
    # vis_relative_position('./data/00109.npy')
