import numpy as np
import os
import cv2
from keras.models import Sequential,load_model
from keras.layers import *
from keras.preprocessing.image import img_to_array
from keras.callbacks import TensorBoard
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.optimizers import Adam


def extract_data(file_path, norm_size):

    image_data = []
    label_data = []

    for data_catalog in os.listdir(file_path):
        for each_image in os.listdir(file_path + '/' + data_catalog):
            img_read = cv2.imread(file_path + '/' + data_catalog + '/' + each_image, cv2.IMREAD_GRAYSCALE)
            img_read = cv2.resize(img_read, (norm_size, norm_size), interpolation=cv2.INTER_CUBIC)
            img_read = img_to_array(img_read)

            image_data.append(img_read)
            label_data.append(data_catalog)

    image_data = np.array(image_data, dtype="float") / 255.0
    label_data = np.array(label_data)
    label_data = to_categorical(label_data, num_classes=2)

    print(image_data.shape)
    print(label_data.shape)
    return image_data, label_data



def build_model_AlexNet(normal_size, class_num):
    # initialize the model
    model = Sequential()
    inputShape = (normal_size, normal_size, 1)

    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(class_num))
    model.add(Activation("softmax"))

    # return the constructed network architecture
    return model

def train_model(model_to_train, image_data, label_data, model_name):
    print("Train-Validation Split")
    train_x, vali_x, train_y, vali_y = train_test_split(image_data, label_data, test_size=0.2)

    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6, verbose=1)
    early_stopper = EarlyStopping(min_delta=0.001, patience=10, verbose=1)
    checkpoint = ModelCheckpoint('./model/' + model_name + '_model.h5',monitor="val_acc", verbose=1,save_best_only=True, save_weights_only=True,mode="max")
    model_to_train.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-3 / 30), metrics=['accuracy'])
    model_to_train.fit(train_x, train_y, batch_size=64,
              epochs=30,
              validation_data=(vali_x, vali_y),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, checkpoint]
              )
    print("Finish training")

def evaluate_model(model_to_evaluate, image_data, label_data):
    loss, accuracy = model_to_evaluate.evaluate(image_data, label_data)

    print('test loss;', loss)
    print('test accuracy:', accuracy)


def predict_image(image_path, model_path, model_name):
    weights_path = model_path + model_name + '_model.h5'
    predict_image_data = []

    for each_image in os.listdir(image_path):
        img_read = cv2.imread(image_path + each_image, cv2.IMREAD_GRAYSCALE)
        img_read = cv2.resize(img_read, (64, 64), interpolation=cv2.INTER_CUBIC)
        img_read = img_to_array(img_read)

        predict_image_data.append(img_read)


    predict_image_data = np.array(predict_image_data, dtype="float") / 255.0

    model_to_predict = build_model_AlexNet(64, 2)
    model_to_predict.load_weights(weights_path)
    result = model_to_predict.predict_proba(predict_image_data)
    print(result)
    for each_result  in result:
        if np.argmax(each_result) == 0:
            print("cat with")
            print(each_result[0])
        else:
            print("dog with")
            print(each_result[1])

