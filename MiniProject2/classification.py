import numpy as np
import os
import cv2
from keras.models import Sequential,load_model
from keras.layers import *
from keras.callbacks import TensorBoard
from keras.utils import plot_model



def extract_data(file_path):

    train_image_raw_data = []
    test_image_raw_data = []
    train_label_data = []
    test_label_data = []

    train_sub_path = '/dataset/train/'
    test_sub_path = '/dataset/test/'

    for data_catalog in os.listdir(file_path + train_sub_path):
        for each_image in os.listdir(file_path + train_sub_path + data_catalog):
            train_image_raw_data.append(cv2.resize(
                cv2.imread(file_path + train_sub_path + data_catalog + '/' + each_image, cv2.IMREAD_GRAYSCALE), (64, 64),
                interpolation=cv2.INTER_CUBIC))
            train_label_data.append(data_catalog)

    for data_catalog in os.listdir(file_path + test_sub_path):
        for each_image in os.listdir(file_path + test_sub_path + data_catalog):
            test_image_raw_data.append(cv2.resize(cv2.imread(file_path + test_sub_path + data_catalog + '/' + each_image, cv2.IMREAD_GRAYSCALE), (64, 64), interpolation=cv2.INTER_CUBIC))
            test_label_data.append(data_catalog)

    train_data_set = np.array([train_image_raw_data, train_label_data])
    train_data_set = train_data_set.transpose()

    test_data_set = np.array([test_image_raw_data, test_label_data])
    test_data_set = test_data_set.transpose()

    x = np.array(train_data_set[:,0])
    print('okok')
    print(x.shape[0])
    print(np.array(x[0]).shape)

    np.random.shuffle(train_data_set)
    np.random.shuffle(test_data_set)

    return train_data_set, test_data_set



def extract_class_num(file_path):
    class_num = 0

    for data_catalog in os.listdir(file_path + '/dataset/test'):
        class_num = class_num + 1

    return class_num

def build_model_CNN_1():
    model = Sequential()

    model.add(InputLayer(input_shape=[64, 64, 1]))
    model.add(Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=50, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=80, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='softmax'))

    model.summary()

    return model

def train_model(model_to_train, train_data):
    x_train = np.array([i[0] for i in train_data]).reshape(-1, 64, 64, 1)
    y_train = np.array([i[1] for i in train_data])
    model_to_train.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model_to_train.fit(x=x_train, y=y_train, epochs=12,batch_size=20,validation_split=0.3,shuffle=True)
    model_to_train.save('./generated_model/')

def evaluate_model(model_to_evaluate, test_data):
    x_test = np.array([i[0] for i in test_data]).reshape(-1, 64, 64, 1)
    y_test = np.array([i[1] for i in test_data])
    loss, accuracy = model_to_evaluate.evaluate(x_test, y_test)
    print('test loss',loss)
    print('test accuracy',accuracy)