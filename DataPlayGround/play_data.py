#This script should load data from any format and convert to the list of data object which contains image data as
#numpy ndarray and labels as integers.

from tensorflow import keras
import glob
import os
import imageio
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing import image

class DataStore:

    def __init__(self,train_data_path="",test_data_path="",seed=42,batch_size=10,target_image_size=448,train_val_split=0.3):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.TrainData = self.ValidData = self.TestData = None
        self.seed = seed
        self.batch_size = batch_size
        self.target_image_size = (target_image_size,target_image_size)
        self.train_val_split = train_val_split
        self.global_class_names = self.GetClassNames()

    def LoadTrainData(self):
        train_data_gen = ImageDataGenerator(featurewise_center=False,
                                            samplewise_center=False,
                                            featurewise_std_normalization=False,
                                            samplewise_std_normalization=False,
                                            zca_whitening=False,
                                            rotation_range=20,  # 0.
                                            width_shift_range=0.2,  # 0.
                                            height_shift_range=0.2,  # 0.
                                            shear_range=0.,
                                            zoom_range=0.,
                                            channel_shift_range=0.,
                                            fill_mode='nearest',
                                            cval=0.,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            preprocessing_function=preprocess_input,
                                            validation_split=self.train_val_split,
                                            data_format=K.image_data_format())
        valid_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            validation_split=self.train_val_split,
                                            data_format=K.image_data_format())
        train_gen = train_data_gen.flow_from_directory(self.train_data_path, subset='training',
                                                       shuffle=True, seed=self.seed, target_size=self.target_image_size,
                                                       batch_size=self.batch_size,
                                                       classes=['0','1'])
        valid_gen = valid_data_gen.flow_from_directory(self.train_data_path, subset='validation',
                                                       shuffle=True, seed=self.seed,
                                                       target_size=self.target_image_size,
                                                       batch_size=self.batch_size,
                                                       classes=['0','1'])
        self.TrainData = train_gen
        self.ValidData = valid_gen
        print("Data is loaded and nornalized")

    def LoadTestData(self):
        test_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                            data_format=K.image_data_format())
        test_gen = test_data_gen.flow_from_directory(self.test_data_path,
                                                     shuffle=False,
                                                     target_size=self.target_image_size,
                                                     batch_size=self.batch_size,
                                                     classes=['0','1'])
        self.TestData = test_gen
        print("Test Data is loaded and nornalized")

    def GetClassNames(self):
        return ['invasive','non-invasive']