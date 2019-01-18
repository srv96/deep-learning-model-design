import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import DeepLearning
import SpecialFunction
import Layer
# Importing the dataset
train_dataset = pd.read_csv('mnist_train_small.csv')
X_train =train_dataset.iloc[:,1:].values
y_train =train_dataset.iloc[:,0].values

test_dataset = pd.read_csv('mnist_test_small.csv')
X_test = test_dataset.iloc[:,1:].values
y_test = test_dataset.iloc[:,0].values

classifier = DeepLearning.NeuralNetwork()
<<<<<<< HEAD
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=40,activation ='sigmoid',initializer = 'gaussian',input_dim=784,random_state = 1))
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=20,activation ='sigmoid',initializer = 'gaussian',random_state = 3))
#classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=10,activation ='sigmoid',initializer = 'gaussian',random_state = 2))
=======
#classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=40,activation ='sigmoid',initializer = 'gaussian',input_dim=784,random_state = 1))
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=20,activation ='sigmoid',initializer = 'gaussian',random_state = 3))
classifier.add(DeepLearning.Layer(neural_network=classifier,output_dim=10,activation ='sigmoid',initializer = 'gaussian',random_state = 2))
>>>>>>> b806196bb6abb48ba45a58373796e88f513d49df
input_row = X_train
target_row = y_train
classifier.fit(X_train = input_row, y_train = target_row,epoch =1000,learning_rate =0.01)
y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
SpecialFunction.confusion_matrix(y_test =y_test,y_pred= y_pred)
