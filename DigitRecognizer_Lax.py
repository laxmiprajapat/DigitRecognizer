# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:26:31 2016

@author: lax
"""

#import libraries
import numpy as np
import pandas as pd
import csv
import scipy

#import CNN libraries
import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

#visualisation libraries
"""
from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
"""

#load the data
data = pd.DataFrame(pd.read_csv('train.csv', sep=',', header=0))
testing = pd.DataFrame(pd.read_csv('test.csv', sep=',', header=0))

headertrain = data.columns.values
headertest = testing.columns.values

print data.shape
print data.head()
print headertrain

# convert the data frame to a numpy array and check the dimensionality
dataArray = np.array(data)
testingArray = np.array(testing)
 
# split the data into X (independent variable) and y (dependent/target variable).
X = dataArray[:, 1:].reshape((-1, 1, 28, 28)).astype(np.uint8)
y = dataArray[:, 0].astype(np.uint8)

testingX = testingArray.reshape((-1, 1, 28, 28)).astype(np.uint8)

yFreq = scipy.stats.itemfreq(y)
print yFreq

#XTrain, XTest, yTrain, yTest = train_test_split(X/255., y, train_size=0.8, random_state=0)

np.random.seed(42)

#neural networks
#our input layer has 784 nodes (total no of pixels)
#learn rate = gradient descent steps (smaller rate = smaller steps)
#epochs = number of iterations (higher is better as it has more time to converge)

CNN = NeuralNet(
            layers=[('input', layers.InputLayer),
            ('conv2d1', layers.Conv2DLayer),
            ('conv2d2', layers.Conv2DLayer),
            ('conv2d3', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('conv2d4', layers.Conv2DLayer),
            ('conv2d5', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),            
            ('dense1', layers.DenseLayer),
            ('dropout1', layers.DropoutLayer),
            ('dense2', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
    
    # input layer
    # "1" because the image is grey scale 
    input_shape=(None, 1, 28, 28),
    
    # 3 convolutional layers - connected to local regions in input image
    conv2d1_num_filters=32,
    conv2d1_filter_size=(5, 5),
    conv2d1_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d1_W=lasagne.init.HeNormal(gain='relu'),  
    
    conv2d2_num_filters=32,
    conv2d2_filter_size=(3, 3),
    conv2d2_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d2_W=lasagne.init.HeNormal(gain='relu'),
    
    conv2d3_num_filters=64,
    conv2d3_filter_size=(3, 3),
    conv2d3_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d3_W=lasagne.init.HeNormal(gain='relu'),
    
    # layer maxpool1
    # max pooling reduces the convolution dimensions to (10, 10)
    maxpool1_pool_size=(2, 2),   
    
    # 2 convolutional layers
    conv2d4_num_filters=96,
    conv2d4_filter_size=(5, 5),
    conv2d4_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d4_W=lasagne.init.HeNormal(gain='relu'),
    
    conv2d5_num_filters=128,
    conv2d5_filter_size=(3, 3),
    conv2d5_nonlinearity=lasagne.nonlinearities.rectify,
    conv2d5_W=lasagne.init.HeNormal(gain='relu'),
    
    # layer maxpool2
    # max pooling further reduces the image size to (2, 2)
    maxpool2_pool_size=(2, 2),
        
    # dense layer 1 - hidden layer is fully connected
    dense1_num_units=512,
    dense1_nonlinearity=lasagne.nonlinearities.rectify,
    dense1_W=lasagne.init.HeNormal(gain='relu'),
    
    # dropout 1
    # dropout layer is a regularizer that randomly sets input values to 0
    dropout1_p=0.45, 
    
    # dense layer 2 - hidden layer is fully connected
    dense2_num_units=512,
    dense2_nonlinearity=lasagne.nonlinearities.rectify, 
    dense2_W=lasagne.init.HeNormal(gain='relu'),
        
    # output
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=10,
    
    # optimization method params
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=20,
    verbose=1,
    )

CNNfinal = CNN.fit((X/255.), y)
CNNfinalpred = CNNfinal.predict(testingX/255.0)

c = 1
with open('CNN_FinalPredictions.csv','w') as f:
    headings = ["ImageId", "Label"]
    writer=csv.writer(f, delimiter=',',lineterminator='\n',)
    writer.writerow(headings)
    for line in range(len(CNNfinalpred)):
        row = [c] + [CNNfinalpred[line]]
        writer.writerow(row)
        c+=1
