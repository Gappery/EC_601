# mini_project_2 Deep Learning: Binary Image Classification
## 1. Project Introduction
In this project, using Keras, which is a open source neural network library written by python, I build two sequential model and evaluate their performance given a specific dataset.
## 2. Prerequisites
### 2.1 Data Preparation
#### 2.1.1 Data Set
Since it is a binary classification project, the requirement for the dataset is that it contains only two types of data and the images are labeled and classified.<br>
[Here is a example data set][https://www.kaggle.com/c/dogs-vs-cats-aca2018/data]
#### 2.1.2 Data Organization
**Note**: <br>If you want to use the project as an API, there is no requirement for the location of the data, you only need to know how to invoke functions correctly<br>
There are mainly four types of data: training, valiation, testing, predicting. But for user, you only need to provide three types, they are training, testing and predicting, the code will split the validation part from trainign data<br>
To make sure the project can run successfully, please put the data to the correct directory<br>
*classfication.py<br>
*main.py<br>
*dataset<br>
*model<br>
*predict<br>
	*images(please put the first type train data here)<br>

*dataset<br>
	*train<br>
	*test<br>
		*type1(modify it to the correct type name)<br>
			*images(please put the first type train data here)<br>
		*type2(modify it to the correct type name)<br>
			*images(please put the first type train data here)<br>


**Note**: All data should be pictures with format "jpg"
### 2.2 Corresponding Package Installation
**Note**: All installations are in windows environment and using pip methods
#### 2.2.1 Keras </br>
```$ pip install keras -U --pre```</br>
#### 2.2.2 opencv-python </br>
```$ pip install opencv-python```</br>
#### 2.2.3 pillow-pil </br>
```$ pip install Pillow```</br>
## 3. Run Project
### 3.1 Run as API
There is the description for function and parameters of every methods in the code file. Generally, here is the basic procedure.<br>
#### 3.1.1 First Step: Data Extraction and Pre-processing
extract_image_mapping
extract_data
#### 3.1.2 Second Step: Model Building and Model Training
build_model_1
build_model_2
train_model
evaluate_model
#### 3.1.3 Last Step: Image Prediction
predict_image
### 3.2 Run as project
**Note**: Make sure you have finished 2.1.2 first
cd to the correct location, then:
```$ \python main.py</br>