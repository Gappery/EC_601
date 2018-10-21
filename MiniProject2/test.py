import os
import tensorflow as tf
import numpy as np

dataset_path = "./" + 'train'

total_image_raw_data = []
total_label_raw_data = []

for dataset_catalog in os.listdir(dataset_path):
    for dataset_raw_image in os.listdir(dataset_path + '/' + dataset_catalog):
        total_image_raw_data.append(dataset_path + '/' + dataset_catalog + '/' + dataset_raw_image)
        total_label_raw_data.append(dataset_catalog)

total_dataset_info = np.array([total_image_raw_data, total_label_raw_data])
total_dataset_info = total_dataset_info.transpose()
np.random.shuffle(total_dataset_info)
print(total_dataset_info[525][0])
print(total_dataset_info[525][1])
test_data_number = (int)(total_dataset_info.shape[0] * 0.8)
test_dataset_info = total_dataset_info[:test_data_number]
test_dataset_images = total_dataset_info[:,0]
test_dataset_labels = total_dataset_info[:,1]


test_dataset_image = tf.cast(test_dataset_image)




"""
resize_w = 64
resize_h = 64
test_images = tf.cast(test_dataset_image, tf.string)
test_labels = tf.cast(test_dataset_label, tf.int64)
queue = tf.train.slice_input_producer([test_images, test_labels])
test_labels = queue[1]
test_images_c = tf.read_file(queue[0])
test_images = tf.image.decode_jpeg(test_images_c, channels=3)
test_images = tf.image.resize_image_with_crop_or_pad(test_images, resize_w, resize_h)
test_images = tf.image.per_image_standardization(test_images)

batch_images, batch_labels = tf.train.batch([test_images, test_labels], batch_size=test_data_number, num_threads=64)
"""
