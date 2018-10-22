import classification

print(classification.extract_class_num('.'))
train_data, test_data = classification.extract_data('.')
train_model = classification.build_model_CNN_1()
classification.train_model(train_model, train_data)
classification.evaluate_model(train_model, test_data)