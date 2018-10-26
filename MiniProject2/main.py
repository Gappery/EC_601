import classification
import os

train_data_path = './dataset/train'
test_data_path = './dataset/test'
model_save_path = './model/'


CLASS_NUM = 2
IMG_NORMAL_SIZE = 64
'''
print("***********************************************************************************************************")
print("*************************First Step: Data Extraction and Pre-processing************************************")
print("***********************************************************************************************************")
print("------------Before extracting the data, we first do a mapping between number and two types")
image_mapping = classification.extract_image_mapping(train_data_path)
print("------------Firstly, we extract the training dataset images")
train_images, train_labels = classification.extract_data(train_data_path, IMG_NORMAL_SIZE, image_mapping)
print("------------Then, we extract the testing dataset images")
test_images, test_labels = classification.extract_data(test_data_path, IMG_NORMAL_SIZE, image_mapping)

print("***********************************************************************************************************")
print("*************************Second Step: Model Building and Model Training************************************")
print("***********************************************************************************************************")
print("------------Building the first model: model_1")
model_1 = classification.build_model_1(IMG_NORMAL_SIZE, CLASS_NUM)
print("------------Building the second model: model_2")
model_2 = classification.build_model_2(IMG_NORMAL_SIZE, CLASS_NUM)
print("------------Train and evaluate model_1")
classification.train_model(model_1, train_images, train_labels, 1)
classification.evaluate_model(model_1, test_images, test_labels)
print("------------Train and evaluate model_2")
classification.train_model(model_2, train_images, train_labels, 2)
classification.evaluate_model(model_2, test_images, test_labels)

print("***********************************************************************************************************")
print("**************************************Last Step: Image Prediction******************************************")
print("***********************************************************************************************************")
'''
image_mapping = classification.extract_image_mapping(train_data_path)
print("------------Image Prediction using model 1:")
classification.predict_image('./predict/', model_save_path, 1, image_mapping)

print("------------Image Prediction using model 2:")
classification.predict_image('./predict/', model_save_path, 2, image_mapping)
