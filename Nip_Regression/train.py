from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
from config import config
from utilities import *
from tqdm import tqdm
from detector import Detector
from utils.train_manager import TrainManager


def tf_get_crop_value(img, coord):

    img_w = tf.cast(tf.shape(img)[1],dtype=tf.float32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.float32)

    x = coord[1]
    y = coord[2]

    x1_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=tf.math.minimum(img_w*0.2, x)), dtype=tf.int32), [])
    x2_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=tf.math.minimum(img_w*0.2, img_w - x)), dtype=tf.int32), [])
    y1_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=tf.math.minimum(img_h*0.2, y)), dtype=tf.int32), [])
    y2_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=tf.math.minimum(img_h*0.2, img_h - y)), dtype=tf.int32), [])


    return [x1_c, x2_c, y1_c, y2_c]


def tf_crop(img, pec, coord):
    coord = coord * tf.constant([1, 256, 512], dtype=tf.float32)

    img_w = tf.cast(tf.shape(img)[1], dtype=tf.int32)
    img_h = tf.cast(tf.shape(img)[0], dtype=tf.int32)
    x1_c, x2_c, y1_c, y2_c = tf_get_crop_value(img, coord)

    coord = tf.math.subtract(coord, [0, x1_c, y1_c])
    img = img[y1_c:img_h-y2_c, x1_c:img_w-x2_c]
    pec = pec[y1_c:img_h-y2_c, x1_c:img_w-x2_c]

    # resize to original size
    coord = tf.math.multiply(coord, [1, 256 / (img_w - x1_c - x2_c), 512 / (img_h - y1_c - y2_c)])
    img = tf.image.resize(img, size=[512, 256])
    pec = tf.image.resize(pec, size=[512, 256])

    coord = tf.math.truediv(coord, tf.constant([1, 256, 512], dtype=tf.float32))

    return img, pec, coord


def flip_horizontal(img, pec, coord):
    do_flip = tf.reshape(tf.random.uniform([1], minval=0, maxval=1), []) > tf.reshape(config.flip_rate, [])
    img = tf.cond(do_flip, lambda:tf.image.flip_left_right(img), lambda: img)
    pec = tf.cond(do_flip, lambda:tf.image.flip_left_right(pec), lambda: pec)
    coord = tf.cond(do_flip,
                    lambda: tf.truediv(tf.math.abs(tf.math.subtract(tf.constant([0, 256, 0], dtype=tf.float32), coord * tf.constant([1, 256, 1], dtype=tf.float32))), tf.constant([1, 256, 1], dtype=tf.float32)),
                    lambda: coord)
    return img, pec, coord


def flip_vertical(img, pec, coord):
    do_flip = tf.reshape(tf.random.uniform([1], minval=0, maxval=1), []) > tf.reshape(config.flip_rate, [])
    img = tf.cond(do_flip, lambda:tf.image.flip_up_down(img), lambda: img)
    pec = tf.cond(do_flip, lambda:tf.image.flip_up_down(pec), lambda: pec)
    coord = tf.cond(do_flip,
                    lambda: tf.truediv(tf.math.abs(tf.math.subtract(tf.constant([0, 0, 512], dtype=tf.float32), coord * tf.constant([1, 1, 512], dtype=tf.float32))), tf.constant([1, 1, 512], dtype=tf.float32)),
                    lambda: coord)
    return img, pec, coord


def parse_func(img_path, pec_path, nip_coord, window):
    nip_coord = tf.cast(nip_coord, dtype=tf.float32)

    img_file = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_file, channels=1, dtype=tf.uint16)
    img = tf.image.resize(img, [512, 256])
    img = tf.cast(img, tf.float32)

    pec_file = tf.io.read_file(pec_path)
    pec = tf.io.decode_png(pec_file, channels=1, dtype=tf.uint16)
    pec = tf.image.resize(pec, [512, 256])
    pec = tf.cast(pec, tf.float32)

    # random flip
    if config.random_flip:
        img, pec, nip_coord = flip_horizontal(img, pec, nip_coord)
        img, pec, nip_coord = flip_vertical(img, pec, nip_coord)

    # random crop
    if config.random_crop:
        img, pec, nip_coord = tf_crop(img, pec, nip_coord)

    # window augmentation
    window_center = window[0]
    window_width = window[1]
    if config.random_win_lvl:
        window_offset = tf.random.uniform([1], minval=(-.1*window_center), maxval=(.1*window_center))
        window_center += window_offset
    min_val, max_val = window_center - window_width / 2, window_center + window_width / 2
    img = tf.clip_by_value(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    img = img * 2 - 1

    pec = tf.clip_by_value(pec, 0, 1)

    nip_coord = tf.reshape(nip_coord, shape=[1, 1, 3])

    # random brightness
    if config.random_brightness:
        img = tf.image.random_brightness(img, .4)

    # same facing
    if config.same_facing:
        left = tf.reduce_sum(img[:, :5, :])
        right = tf.reduce_sum(img[:, -5:, :])

        img = tf.cond(tf.math.less(left, right), lambda: tf.image.flip_left_right(img), lambda: img)
        pec = tf.cond(tf.math.less(left, right), lambda: tf.image.flip_left_right(pec), lambda: pec)
        nip_coord = tf.cond(tf.math.less(left, right),
                            lambda: tf.math.abs(tf.math.subtract(tf.constant([0, 256, 0], dtype=tf.float32), nip_coord)),
                            lambda: nip_coord)
    return img, pec, nip_coord


def build_input_pipeline(batch_size, img_file_path_list, pec_file_path_list, nip_coord_list, window_list):
    ds_train = tf.data.Dataset.from_tensor_slices((img_file_path_list, pec_file_path_list, nip_coord_list, window_list))
    ds_train = ds_train.shuffle(500000)
    ds_train = ds_train.map(parse_func, num_parallel_calls=12)

    ds_train = ds_train.repeat().batch(batch_size).prefetch(batch_size * 3)   # add shuffling

    return ds_train


def get_data_list(data_folders):
    train_img_list = []
    train_pec_list = []
    train_nip_coord_list = []
    train_window_list = []

    test_img_list = []
    test_pec_list = []
    test_nip_coord_list = []
    test_window_list = []
    for data_folder in data_folders:
        for root, dirs, files in os.walk(data_folder):
            for name in files:
                if '.png' in name and 'Img_Crop' in root and 'None' not in name:
                    # print(name)
                    if name in config.coord_dict and os.path.exists(os.path.join(config.pec_label_folder, name)):
                        split = config.train_test_dict[name]
                        img_path = os.path.join(root, name)
                        pec_path = os.path.join(config.pec_label_folder, name)
                        if split < config.train_portion:
                            train_nip_coord_list.append(config.coord_dict[name])
                            train_img_list.append(img_path)
                            train_pec_list.append(pec_path)
                            train_window_list.append(config.info_dict[img_path.split('/')[-1][:-4]])
                        else:
                            test_nip_coord_list.append(config.coord_dict[name])
                            test_img_list.append(img_path)
                            test_pec_list.append(pec_path)
                            test_window_list.append(config.info_dict[img_path.split('/')[-1][:-4]])

    train_set = (train_img_list, train_pec_list, train_nip_coord_list, train_window_list)
    test_set = (test_img_list, test_pec_list, test_nip_coord_list, test_window_list)

    return train_set, test_set


if __name__ == '__main__':
    # print(coord_dict)
    train, test = get_data_list(config.data_folders)
    train_img_list, train_pec_list, train_nip_coord_list, train_window_list = train
    test_img_list, test_pec_list, test_nip_coord_list, test_window_list = test
    # print(img_list)
    train_dataset = build_input_pipeline(config.batch_size, train_img_list, train_pec_list, train_nip_coord_list, train_window_list)
    test_dataset = build_input_pipeline(config.batch_size, test_img_list, test_pec_list, test_nip_coord_list, test_window_list)

    train_manager = TrainManager(Detector, train_dataset, test_dataset)
    train_manager.fit()
