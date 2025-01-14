# -*- coding: utf-8 -*-
"""Untitled13.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GC1mYxJgBA4V7ZmgVjKzJ8c7cZ-benoV
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class logistic_reg():
    
    def __init__(self, learning_rate = 0.00001, no_of_iteration = 700, concerned_class = 0):
        
        # Initializing Values!
        
        self.learning_rate = learning_rate
        self.no_of_iteration = no_of_iteration
        self.concerned_class = concerned_class
        self.no_of_example = 0
        self.cst = 0
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
        
    
    def cost_func(self, y_pred, Y_train):
     
        # Using 'One Vs. All method'!
        # Labelling the 'concerned_class' target values as '1' and all other classes as '0'.
        Y = [1 if Y_train[i] == self.concerned_class else 0 for i in range(len(Y_train))]
        Y = np.array(Y)
        for i in range(len(y_pred)):
          if y_pred[i]==1:
            y_pred[i] -= 1e-4   
        cost = ( -1*sum( Y*np.log(y_pred) + (1-Y)*np.log(1-y_pred) ) )/self.no_of_example
        return cost
    
    def mini_batches(self, X, Y, batch_size):
        
        #Making mini batches!
        
      np.random.seed(1)
      np.random.shuffle(X)
      np.random.seed(1)
      np.random.shuffle(Y)
      indices = np.arange(X.shape[0])
      for ind in range(0, X.shape[0], batch_size):
        last_ind = min(ind + batch_size, X.shape[0])
        index = indices[ind:last_ind]
        yield X[index], Y[index]
        
    def train(self, X_train, Y_train, batch_size = 50):
        ''' This method trains the algorithm on the given dataset!
            It takes parameters as training input and training output labels and batch size(optional).'''
        
        self.no_of_example, no_of_feature = X_train.shape
        
        # Initializing constant and weight to be passed as a parameter to 'sigmoid()'!
        
        self.const = 0
        self.weight = [0 for i in range(no_of_feature)]
        Y = [1 if Y_train[i] == self.concerned_class else 0 for i in range(self.no_of_example)]
        Y = np.array(Y)
        
        # Making cst's dimension equal to the number of iterations!
        
        self.cst = np.zeros(self.no_of_iteration)
        
        # Applying Gradient Descent.
        
        for i in range(self.no_of_iteration): 
          for batches in self.mini_batches(X_train, Y, batch_size):
            x_train, y_train = batches       
            z = self.const + np.dot(x_train, self.weight)
            y_pred = self.sigmoid(z)
            
            # Calculating gradients with respect to constant and weights.
            
            dc = sum(y_pred - y_train)
            dw = np.dot(x_train.T, (y_pred - y_train))
            
            # Updating parameters.
            
            self.const -= (self.learning_rate / self.no_of_example)*dc
            self.weight -= (self.learning_rate / self.no_of_example)*dw   
            self.cst[i] = self.cost_func(y_pred, y_train)  
        
    def predict(self, X_test):
        '''This method predicts the output label as 1 if this is the concerned class else 0!'''
        
        # This function predicts the output corresponding to the given test data.
        
        z = self.const + np.dot(X_test, self.weight)
        pred = self.sigmoid(z)
        
        # Predict the class as positive class if 'pred' gives a value greater than '0.5',
        # else predict the class as negative class!
        
        Y_pred = [1 if i >= 0.5 else 0 for i in pred]
        Y_pred = np.array(Y_pred)
        return Y_pred
    
    def accuracy(self, X_test, Y_test):
        
        # This function calculates the accuracy of predicted values.
        
        sz = len(Y_test)
        sum = 0
        Y_pred = self.predict(X_test)
        Y = [1 if Y_test[i] == self.concerned_class else 0  for i in range(sz)]
        Y = np.array(Y)
        for i in range(sz):
            if Y[i] == Y_pred[i]:
                sum += 1
        return (sum/sz)*100
        
    def learning_curve(self):
        
        # This function plots a learning curve viz. plot of 'Number of iterations' Vs. 'Cross 
        # Entropy Loss'.
    
        iter = [(i + 1) for i in range(self.no_of_iteration)]
        iter = np.array(iter)
        plt.plot(iter, self.cst, label = self.concerned_class)
        plt.title('Learning Curve')
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost Entropy Loss')
        plt.legend(title = 'Classes', fancybox = True, shadow = True, ncol = 2, borderpad = 1)
        plt.show