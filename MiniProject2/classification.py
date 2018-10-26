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
import Image
import ImageDraw
import ImageFont


# This method returns the mapping between images and their types
def extract_image_mapping(file_path):

    map_catagory = {}
    image_catagory = 0

    for data_catalog in os.listdir(file_path):
        map_catagory[data_catalog] = image_catagory
        image_catagory = image_catagory + 1


    print("image-number mapping finished, they are:")
    print(map_catagory)
    return map_catagory



# The function extracts images, preprocesses images and labels those images, it returns the array with processed image data and corresponding label data
def extract_data(file_path, norm_size, image_mapping):
    image_data = []
    label_data = []

    for data_catalog in os.listdir(file_path):
        print("Extracting data with type: " + data_catalog)

        for each_image in os.listdir(file_path + '/' + data_catalog):
            img_read = cv2.imread(file_path + '/' + data_catalog + '/' + each_image, cv2.IMREAD_GRAYSCALE)
            img_read = cv2.resize(img_read, (norm_size, norm_size), interpolation=cv2.INTER_CUBIC)
            img_read = img_to_array(img_read)

            image_data.append(img_read)
            label_data.append(image_mapping[data_catalog])

    print("Extraction finished, now doing the preprocessing")
    image_data = np.array(image_data, dtype="float") / 255.0
    label_data = np.array(label_data)
    label_data = to_categorical(label_data, num_classes=2)

    print(image_data.shape)
    print(label_data.shape)
    return image_data, label_data


# The function builds and returns the first model
def build_model_1(normal_size, class_num):
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

# The function builds and returns the first model
def build_model_1(normal_size, class_num):
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
# The function builds and returns the second model
def build_model_2(normal_size, class_num):
    model = Sequential()
    inputShape = (normal_size, normal_size, 1)

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_num))
    model.add(Activation('softmax'))

    return model


#  The function trains the data for a specific model, it splits the training data into training part and validation part
def train_model(model_to_train, image_data, label_data, model_name):
    print("Train-Validation Split")
    train_x, vali_x, train_y, vali_y = train_test_split(image_data, label_data, test_size=0.2)
    # set reduce parameters
    lr_reducer = ReduceLROnPlateau(factor=0.005, cooldown=0, patience=5, min_lr=0.5e-6, verbose=1)
    # set early_stop parameters to early stop the function when there is no improvement with the model
    early_stopper = EarlyStopping(min_delta=0.001, patience=10, verbose=1)
    # save the best model
    checkpoint = ModelCheckpoint('./model/' + str(model_name) + '_model.h5',monitor="val_acc", verbose=1,save_best_only=True, save_weights_only=True,mode="max")
    model_to_train.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, decay=1e-3 / 30), metrics=['accuracy'])
    model_to_train.fit(train_x, train_y, batch_size=64,
              epochs=20,
              validation_data=(vali_x, vali_y),
              shuffle=True,
              callbacks=[lr_reducer, early_stopper, checkpoint]
              )
    print("Finish training")

#  The function evaluates the accuracy for the trained model
def evaluate_model(model_to_evaluate, image_data, label_data):
    loss, accuracy = model_to_evaluate.evaluate(image_data, label_data)

    print('test loss;', loss)
    print('test accuracy:', accuracy)


#  The function predicts the unlabeled image using a specific model
def predict_image(image_path, model_path,  model_type, image_mapping):
    print(image_mapping)
    image_mapping = dict(zip(image_mapping.values(), image_mapping.keys()))
    print(image_mapping)
    weights_path = model_path + str(model_type) + '_model.h5'

    print("model type: " + str(model_type))
    if model_type == 1:
        model_to_predict = build_model_1(64, 2)
    else:
        model_to_predict = build_model_2(64, 2)

    model_to_predict.load_weights(weights_path)

    font = ImageFont.truetype('simsun.ttc', 10)
    print("Start prediction..................")
    for each_image in os.listdir(image_path):
        predict_image_data = []
        img_read = cv2.imread(image_path + each_image, cv2.IMREAD_GRAYSCALE)
        img_read = cv2.resize(img_read, (64, 64), interpolation=cv2.INTER_CUBIC)
        img_read = img_to_array(img_read)

        predict_image_data.append(img_read)
        predict_image_data = np.array(predict_image_data, dtype="float") / 255.0

        result = model_to_predict.predict_proba(predict_image_data)

        im = Image.open(image_path + each_image)
        draw = ImageDraw.Draw(im)
        if np.argmax(result) == 0:
            if model_type == 1:
                text_to_draw = "Model 1 predicts that it belongs to type: " + image_mapping[0] + ". The probability is:" + str(result[0][0])
                draw.text((20, 20), text=text_to_draw, fill=(255, 0, 0), font=font)
            else:
                text_to_draw = "Model 2 predicts that it belongs to type: " + image_mapping[0] + ". The probability is:" + str(result[0][0])
                draw.text((20, 35), text=text_to_draw, fill=(0, 0, 255), font=font)

            im.save(image_path + each_image)
        else:
            if model_type == 1:
                text_to_draw = "Model 1 predicts that it belongs to type: " + image_mapping[1] + ". The probability is:" + str(result[0][1])
                draw.text((20, 50), text=text_to_draw, fill=(255, 0, 0), font=font)
            else:
                text_to_draw = "Model 2 predicts that it belongs to type: " + image_mapping[1] + ". The probability is:" + str(result[0][1])
            draw.text((20, 65), text=text_to_draw, fill=(0, 0, 255), font=font)
            im.save(image_path + each_image)

    print("Prediction and label work finished")








