from utilities import *
import argparse
import os


def get_config():
    parser = argparse.ArgumentParser()

    # device
    parser.add_argument('--controller', default='cpu')
    parser.add_argument('--gpu_list', default='0')

    # model params
    # parser.add_argument('--input_shape', default='512,512,3')

    # train control
    parser.add_argument('--total_epoch', default=25)
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--train_portion', default=.9)

    parser.add_argument('--same_facing', default=False)
    parser.add_argument('--random_crop', default=True)
    parser.add_argument('--random_win_lvl', default=True)
    parser.add_argument('--random_flip', default=True)
    parser.add_argument('--flip_rate', default=.5)
    parser.add_argument('--random_brightness', default=True)

    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--model_dir', default='logs/trained_models')

    parsed, unknown = parser.parse_known_args()

    return parsed


config = get_config()
config.info_dict = get_info_dict('DN_Master_0_10_latest.csv')
config.coord_dict = get_coord_dict('nipple_labels.csv')
config.train_test_dict = get_train_test_dict('splited_data.csv')

config.data_folders = ['/data/InHouse_Datasets/Mammo_GAN_DN_Data/DEDUCE_Negative_d_2',
                       '/data/InHouse_Datasets/Mammo_GAN_DN_Data/DEDUCE_Negative_d_4',
                       '/data/InHouse_Datasets/Mammo_GAN_DN_Data/DEDUCE_Negative_d_8']
config.pec_label_folder = '/data/InHouse_Datasets/pec_labels'

# config.input_shape = [int(x) for x in config.input_shape.split(',')]
config.num_gpu = len(config.gpu_list.split(','))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_list
