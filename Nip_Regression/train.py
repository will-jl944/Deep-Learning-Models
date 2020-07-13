from __future__ import print_function
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time, os
from config import config
from utilities import *
import skimage.io
from tqdm import tqdm
from unet import UNet
from regressor import Regressor
from utils.train_manager import Train_manager
import random
import skimage.io


def tf_get_crop_value(img):

    img_w = tf.cast(tf.shape(img)[1],dtype=tf.float32)
    img_h = tf.cast(tf.shape(img)[0],dtype=tf.float32)

    x1_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=img_w*0.05),dtype=tf.int32),[])
    x2_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=img_w*0.05),dtype=tf.int32),[])
    y1_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=img_h*0.05),dtype=tf.int32),[])
    y2_c = tf.reshape(tf.cast(tf.random.uniform([1], minval=0, maxval=img_h*0.05),dtype=tf.int32),[])

    return [x1_c, x2_c, y1_c, y2_c]


def tf_crop(img, pec):
    img_w = tf.cast(tf.shape(img)[1], dtype=tf.int32)
    img_h = tf.cast(tf.shape(img)[0], dtype=tf.int32)
    x1_c, x2_c, y1_c, y2_c = tf_get_crop_value(img)

    img = img[y1_c:img_h-y2_c, x1_c:img_w-x2_c]
    pec = pec[y1_c:img_h-y2_c, x1_c:img_w-x2_c]

    return img, pec


def flip(img, pec):
    do_flip = tf.reshape(tf.random.uniform([1], minval=0, maxval=1),[]) > tf.reshape(config.flip_rate, [])
    img = tf.cond(do_flip, lambda:tf.image.flip_left_right(img), lambda: img)
    pec = tf.cond(do_flip, lambda:tf.image.flip_left_right(pec), lambda: pec)
    return img, pec


def parse_func(img_path, pec_path, nip_coord, window):
    img_file = tf.io.read_file(img_path)
    img = tf.io.decode_png(img_file, channels=1, dtype=tf.uint16)
    img = tf.image.resize(img, [512, 256])
    img = tf.cast(img, tf.float32)

    pec_file = tf.io.read_file(pec_path)
    pec = tf.io.decode_png(pec_file, channels=1, dtype=tf.uint16)
    pec = tf.image.resize(pec, [512, 256])
    pec = tf.cast(pec, tf.float32)

    min_val, max_val = window[0], window[1]
    img = tf.clip_by_value(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)

    pec = tf.clip_by_value(pec, 0, 1)

    if config.same_facing:
        left = tf.reduce_sum(pec[:, :5, :])
        right = tf.reduce_sum(pec[:, -5:, :])

        img = tf.cond(tf.math.less(left, right), lambda: tf.image.flip_left_right(img), lambda: img)
        pec = tf.cond(tf.math.less(left, right), lambda: tf.image.flip_left_right(pec), lambda: pec)
        nip_coord = tf.cond(tf.math.less(left, right),
                            lambda: tf.math.abs(tf.math.subtract(tf.constant([256, 0]), nip_coord)),
                            lambda: nip_coord)
    return img, pec, nip_coord


def build_input_pipeline(batch_size, img_file_path_list, pec_file_path_list, nip_coord_list, window_list):
    ds_train = tf.data.Dataset.from_tensor_slices((img_file_path_list, pec_file_path_list, nip_coord_list, window_list))
    ds_train = ds_train.shuffle(500000)
    ds_train = ds_train.map(parse_func, num_parallel_calls=12)

    ds_train = ds_train.repeat().batch(batch_size).prefetch(batch_size * 3) # add shuffling

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

    train = (train_img_list, train_pec_list, train_nip_coord_list, train_window_list)
    test = (test_img_list, test_pec_list, test_nip_coord_list, test_window_list)

    return train, test


if __name__ == '__main__':
    # print(coord_dict)
    train, test = get_data_list(config.data_folders)
    train_img_list, train_pec_list, train_nip_coord_list, train_window_list = train
    test_img_list, test_pec_list, test_nip_coord_list, test_window_list = test
    # print(img_list)
    train_dataset = build_input_pipeline(config.batch_size, train_img_list, train_pec_list, train_nip_coord_list, train_window_list)
    test_dataset = build_input_pipeline(config.batch_size, test_img_list, test_pec_list, test_nip_coord_list, test_window_list)

    train_manager = Train_manager(Regressor, train_dataset, test_dataset)
    train_manager.fit()
