from __future__ import absolute_import, division, print_function, unicode_literals

#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from DataPlayGround.play_data import DataStore
import math
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
#import from modules
from DataPlayGround.look_at_images import show_figures,plot_image,plot_value_array
import argparse
import configparser
import os
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from utils import symmetric_cross_entropy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import CSVLogger
import shutil

class NeuralNetwork:

    def __init__(self,configParser):
        train_data_path = configParser.get('paths', 'train_data_path')
        test_data_path = configParser.get('paths', 'test_data_path')
        self.operation = configParser.get('action', 'operation')

        seed = configParser.getint('model', 'seed')
        batch_size = configParser.getint('model', 'batch_size')
        train_val_split = configParser.getfloat('model', 'train_val_split')
        target_image_size = configParser.getint('model', 'target_image_size')

        self.target_image_size = target_image_size
        self.batch_size = batch_size
        self.test_data_path = test_data_path

        self.data_store = DataStore(train_data_path=train_data_path,
                                    test_data_path=test_data_path,
                                    seed=seed,
                                    batch_size=batch_size,
                                    train_val_split=train_val_split,
                                    target_image_size=target_image_size)

        self.epochs = configParser.getint('model', 'epochs')
        self.lr = configParser.getfloat('model', 'lr')

        if(self.operation == "train"):
            current_instance = time.strftime("%Y%m%d_%H%M%S")
            self.model_folder = os.path.join(configParser.get('paths', 'model_folder'),current_instance)
            self.graph_folder = os.path.join(configParser.get('paths', 'graph_folder'),current_instance)
            self.log_folder = os.path.join(configParser.get('paths', 'log_folder'), current_instance)
            os.makedirs(self.model_folder)
            os.makedirs(self.graph_folder)
            os.makedirs(self.log_folder)

    def create_model(self):
        img_shape = (self.target_image_size, self.target_image_size, 3)
        # load pre-trained resnet50
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=img_shape))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.50)(x)
        x = Dense(2, activation='softmax', name='fc8')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer=SGD(lr=self.lr, momentum=0.9), loss=symmetric_cross_entropy(1.0,0.1),
                               metrics=['accuracy'])
        if((self.operation == 'test') or (self.operation == 'predict')):
            for layer in model.layers:
                layer.trainable = False
        # Display the model's architecture
        model.summary()
        return model

    def train_model(self):
        model = self.create_model()
        self.data_store.LoadTrainData()
        log_file = os.path.join(self.log_folder, "log.csv")
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto',
                                      min_delta=0.0001,
                                      cooldown=0, min_lr=0.00001)
        checkpoint_path = os.path.join(self.model_folder,"cp-{epoch:04d}.ckpt")
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True,
            period=2)
        csv_logger = CSVLogger(log_file, append=True, separator=';')
        history = model.fit_generator(self.data_store.ValidData,
                                      validation_data=self.data_store.ValidData,
                                      epochs=self.epochs,
                                      callbacks=[earlystop, reduce_lr, cp_callback,csv_logger])
        print("Early stopping stopped after ",earlystop.stopped_epoch)
        model.save_weights(os.path.join(self.model_folder,"final.ckpt"))
        self.plot_training(history)

    def evaluate_model(self,trained_model_path):
        model = self.create_model()
        model.load_weights(trained_model_path)
        self.data_store.LoadTestData()
        test_data = self.data_store.TestData
        steps_ = math.ceil(len(test_data.classes)*1.0/self.batch_size)
        predict = model.predict_generator(test_data, steps=steps_, verbose=1)
        pred_labels = predict.argmax(axis=1)
        test_labels = test_data.classes
        cr_labels = test_data.class_indices
        print(cr_labels)
        print(confusion_matrix(test_labels, pred_labels))
        print(classification_report(test_labels, pred_labels, target_names=cr_labels))
        print(accuracy_score(test_labels, pred_labels))

    def predict(self,trained_model_path):
        model = self.create_model()
        model.load_weights(trained_model_path)
        self.data_store.LoadTestData()
        test_data = self.data_store.TestData
        x,y = test_data.next()
        predict = model.predict_generator(test_data, steps=1, verbose=1)
        gold_labels = y.argmax(axis=1)
        class_names = self.data_store.global_class_names
        #Let's plot several images with their predictions.
        num_rows = 3
        num_cols = 3
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(i, predict[i], gold_labels, x, class_names)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(i, predict[i], gold_labels)
        plt.tight_layout()
        plt.show()

    #Takes data from test_data_path to predict
    def predict_invasive_patches(self,trained_model_path,output_dir):
        model = self.create_model()
        model.load_weights(trained_model_path)
        self.data_store.LoadTestData()
        test_data = self.data_store.TestData
        steps_ = math.ceil(len(test_data.classes) * 1.0 / self.batch_size)
        predict = model.predict_generator(test_data, steps=steps_, verbose=1)
        pred_labels = predict.argmax(axis=1)
        image_names = test_data.filenames
        for i in range(0,len(pred_labels)):
            if(pred_labels[i]==0): #if invasive save it to dir
                path = os.path.join(self.test_data_path,image_names[i])
                shutil.copy(path, output_dir)

    def visualize_images(self,data_name):
        if(data_name == "train"):
            self.data_store.LoadTrainData()
            show_figures(self.data_store.TrainData,self.data_store.global_class_names,20)
        elif(data_name == "valid"):
            self.data_store.LoadTrainData()
            show_figures(self.data_store.ValidData,self.data_store.global_class_names,20)
        else:
            self.data_store.LoadTestData()
            show_figures(self.data_store.TestData, self.data_store.global_class_names,20)

    #Plot the training and validation loss + accuracy
    def plot_training(self,history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc_graph = os.path.join(self.graph_folder,"training_val_acc.png")
        loss_graph = os.path.join(self.graph_folder,"training_val_loss.png")

        epochs = range(len(acc))
        plt.plot(epochs, acc, 'r.')
        plt.plot(epochs, val_acc, 'r')
        plt.title('Training & Validation accuracy')
        plt.show()
        plt.savefig(acc_graph)

        plt.figure()
        plt.plot(epochs, loss, 'r.')
        plt.plot(epochs, val_loss, 'r-')
        plt.title('Training & Validation loss')
        plt.show()
        plt.savefig(loss_graph)

def action(configParser):
    neural_network = NeuralNetwork(configParser)
    operation = configParser.get('action', 'operation')
    if(operation == "train"):
        neural_network.train_model()
    elif(operation == "test"):
        trained_model_path = configParser.get('paths', 'trained_model_path')
        neural_network.evaluate_model(trained_model_path)
    elif(operation == "predict"):
        trained_model_path = configParser.get('paths', 'trained_model_path')
        neural_network.predict(trained_model_path)
    elif (operation == "predict_invasive_patches"):
        trained_model_path = configParser.get('paths', 'trained_model_path')
        output_patches_dir = configParser.get('paths', 'output_patches_dir')
        neural_network.predict_invasive_patches(trained_model_path,output_patches_dir)
    elif(operation == "visualize"):
        data_to_visualize = configParser.get('action', 'data_to_visualize')
        neural_network.visualize_images(data_to_visualize)
    else:
        print("Give proper operation name")

print("Let us do some adventure")
parser = argparse.ArgumentParser(description='Deep Neural Networks')
parser.add_argument('--config', default='config.txt', help='path to config file')
configParser = configparser.RawConfigParser()
configParser.read(parser.parse_args().config)

action(configParser)





