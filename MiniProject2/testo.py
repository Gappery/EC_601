import classification
import os

train_data_path = './dataset/train'
test_data_path = './dataset/test'
model_save_path = './model/'


CLASS_NUM = 2
IMG_NORMAL_SIZE = 64

train_images, train_labels = classification.extract_data(train_data_path, IMG_NORMAL_SIZE)
test_images, test_labels = classification.extract_data(test_data_path, IMG_NORMAL_SIZE)
model_AlexNet = classification.build_model_AlexNet(IMG_NORMAL_SIZE, CLASS_NUM)
classification.train_model(model_AlexNet, train_images, train_labels, 'AlexNet')
classification.evaluate_model(model_AlexNet, test_images, test_labels)
classification.predict_image('./predict/',model_save_path, 'AlexNet' )
#print(classification.extract_class_num('.'))
#train_image_data, train_label_data = classification.extract_data('.')
#train_model = classification.build_model_CNN_1()
#classification.train_model(train_model, train_image_data, train_label_data)
#classification.evaluate_model(train_model, test_data)