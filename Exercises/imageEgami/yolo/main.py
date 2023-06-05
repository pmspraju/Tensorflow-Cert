import argparse
import os
import numpy as np
import json
from parseAnnotation import parse_voc_annotation
from prepareData import create_training_instances

# from yolo import create_yolov3_model, dummy_loss
# from generator import BatchGenerator
# from utils.utils import normalize, evaluate, makedirs
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.optimizers import Adam
# from callbacks import CustomModelCheckpoint, CustomTensorBoard
# from utils.multi_gpu_model import multi_gpu_model
import tensorflow as tf
import keras
from keras.models import load_model

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    # device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def _main_(args):

    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations
    ###############################
    print(config['train']['train_image_folder'])
    print(config['train']['train_annot_folder'])
    print(config['train']['cache_name'])
    print(config['valid']['valid_annot_folder'])
    print(config['valid']['valid_image_folder'])
    print(config['valid']['cache_name'])
    print(config['model']['labels'])

    train_ints, valid_ints, labels, max_box_per_image = create_training_instances(
        config['train']['train_annot_folder'],
        config['train']['train_image_folder'],
        config['train']['cache_name'],
        config['valid']['valid_annot_folder'],
        config['valid']['valid_image_folder'],
        config['valid']['cache_name'],
        config['model']['labels']
    )
    print('\nTraining on: \t' + str(labels) + '\n')

if __name__ == '__main__':

    configPath = r'C:\Users\pmspr\Documents\Machine Learning\Courses\Tensorflow Cert\Git\Tensorflow-Cert\Exercises\imageEgami\yolo'
    configPath = os.path.join(configPath, 'config.json')

    argparser = argparse.ArgumentParser(description='train and evaluate YOLO_v3 model on any dataset')
    argparser.add_argument('-c', '--conf', help='path to configuration file',
                           default=configPath)
    args = argparser.parse_args()
    _main_(args)