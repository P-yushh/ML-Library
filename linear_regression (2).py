# -*- coding: utf-8 -*-
"""Linear Regression.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1whgSF1s_dHSunJDn4n1loQWx1ne8Zace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class linear_reg:
    
    def __init__(self, learning_rate = 1e-7, no_of_iteration = 5000):
        
        # Initializing Values!

        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
        self.no_of_example = 0
        self.mse = 0
    
    def data_split(self, X, Y, test_size = 0.2, cross_validation_size = 0.0, random_state = 1):
        ''' This method can split your data into train, test and cross-validation data according to the input size provided!
               If you don't want cross-validation set then you need not provide its size!'''
        row, col = X.shape
        if type(X) == pandas.core.frame.DataFrame:
            X = X.to_numpy()
        if type(Y) == pandas.core.frame.DataFrame:
            Y = Y.to_numpy()
        # Shuffling the Input data!

        np.random.seed(random_state)
        np.random.shuffle(X)
        np.random.seed(random_state)
        np.random.shuffle(Y)
                
        # Splitting X data!

        test_len = int(row * test_size)
        cross_validation_len = int(row * cross_validation_size)
        temp = np.random.choice(row, test_len, replace = False)
        xtest = X[temp,:]
        X = np.delete(X, (temp), axis = 0)
        temp2 = np.random.choice(row, cross_validation_len, replace = False)
        xvalidation = X[temp2,:]
        X = np.delete(X, (temp2), axis = 0)
        xtrain = X

        # Splitting Y data!

        ytest = Y[temp]
        Y = np.delete(Y, (temp), axis = 0)
        yvalidation = Y[temp2]
        Y = np.delete(Y, (temp2), axis = 0)
        ytrain = Y
        
        # If user wants cross validation then return it, else don't!

        if cross_validation_size == 0:
            return xtrain, xtest, ytrain, ytest
        else:
            return xtrain, xtest, xvalidation, ytrain, ytest, yvalidation    

    def train(self, X_train, Y_train):
        ''' Train the model by providing training input and training labels respectively!'''
        if type(X_train) == pandas.core.frame.DataFrame:
            X_train = X_train.to_numpy()
        if type(Y_train) == pandas.core.frame.DataFrame:
            Y_train = Y_train.to_numpy()
        self.no_of_example, no_of_feature = X_train.shape

        # Initializing constant and weight!

        self.const = 0
        self.weight = [0 for i in range(no_of_feature)]

        # Making mse's dimension equal to the number of iterations!

        self.mse = np.zeros(self.no_of_iteration)

        # Applying Gradient Descent!
        
        for i in range(self.no_of_iteration):
            y_pred = np.dot(X_train, self.weight) + self.const
            
            # Calculating gradients with respect to constant and weights.

            dw = (1/self.no_of_example)*np.dot(X_train.T, (y_pred - Y_train))
            dc = (1/self.no_of_example)*np.sum((y_pred -  Y_train))

            # Updating parameters.

            self.weight -= self.learning_rate*dw
            self.const -= self.learning_rate*dc
            self.mse[i] = self.cost_func(y_pred, Y_train) 
                  
    def predict(self, X_test, limit = 0.9):
        ''' This method predicts the output for the given input and takes input data as parameter, you can also choose to give
                change the value of limit parameter in which case it will alter the accuracy of the predicted output!
                It has already been optimised for the best possible outcome!'''
        
        if type(X_test) == pandas.core.frame.DataFrame:
            X_test = X_test.to_numpy()
        Y_pred = np.dot(X_test, self.weight) + self.const
        Y_pred = self.threshold(Y_pred, limit)
        return Y_pred
    
    def cost_func(self, y_pred, Y_train):
        loss = sum(((y_pred - Y_train)**2)/(2*self.no_of_example))
        return loss
        
    def threshold(self, Y_pred, limit):
        
        # Converting the regression data to classification data! 
        
        for i in range(len(Y_pred)):
                Y_pred[i] = self.explicit_round_off(Y_pred[i], limit)
        return Y_pred
    
    def explicit_round_off(self, float_value, limit):
        if limit == 0:
            raise ValueError('Round Off limit cannot be "0"')
        if float_value >= int(float_value) + limit:
            float_value = int(float_value) + 1
        else:
            float_value = int(float_value)
        return float_value       
        
    def accuracy(self, Y_pred, Y_true):
        
        # Computing accuracy of the predictions made!
        
        sum = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                sum += 1
        return (sum / len(Y_pred)) * 100        
    
    def learning_curve(self):

             # This function plots a learning curve viz. plot of 'Number of iterations' Vs.
             # 'Mean Square Error'.

            iter = [(i + 1) for i in range(self.no_of_iteration)]
            iter = np.array(iter)
            plt.plot(iter, self.mse)
            plt.xlabel('Number of iterations')
            plt.ylabel('Mean Square Error')
            plt.show()
    
    def plot(self, X_test, Y_test):
        ''' This method can plot given data Vs. real output and predicted output provided there are two features in the given input!''' 
            Y_pred = self.predict(X_test)
            plt.scatter(X_test, Y_test)
            plt.plot(X_test, Y_pred, color = 'red')
            plt.xlabel('Test data')
            plt.ylabel('Predicted Output')
            plt.show()
