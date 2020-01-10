#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[22]:


#!/usr/bin/env python
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# In[22]:


class neural_network():
    
    
    def __init__(self, learning_rate = 0.0001, no_of_iteration = 1000, hidden_layers = 5, no_of_neurons = 5, output_nodes = 4):
        self.no_of_iterations = no_of_iteration
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.no_of_neurons = no_of_neurons
        self.output_nodes = output_nodes
        self.weights = []
        self.weights_hidden_layer = []
        self.weight_intermediate_hidden_layer = []
        self.bias = []
        self.bias_hidden_layer = []
        self.uni = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))
    
    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1 - self.sigmoid(z))
    
    def train(self, X, Y):
        np.random.seed(1)
        self.uni = np.unique(Y)
        self.input_weights = np.random.rand(self.no_of_neurons, X.shape[1]+1)*2*0.12 - 0.12
        self.hidden_weights = np.random.rand(self.no_of_neurons, self.no_of_neurons+1, self.hidden_layers -1)*2*0.12 - 0.12
        self.output_weights = np.random.rand(self.output_nodes, self.no_of_neurons+1)*2*0.12 - 0.12
        activation_values = np.zeros([self.no_of_neurons, self.hidden_layers])
        biased_activation_values = np.zeros([self.no_of_neurons+1, self.hidden_layers])
        error_matrix = np.zeros([self.no_of_neurons+1, self.hidden_layers])
        delta_input_weights = np.zeros([self.no_of_neurons, X.shape[1]+1])
        delta_hidden_weights = np.zeros([self.no_of_neurons, self.no_of_neurons+1, self.hidden_layers -1])
        delta_output_weights = np.zeros([self.output_nodes, self.no_of_neurons+1])
        D_input = np.zeros([self.no_of_neurons, X.shape[1]+1])
        D_hidden = np.zeros([self.no_of_neurons, self.no_of_neurons+1, self.hidden_layers -1])
        D_output = np.zeros([self.output_nodes, self.no_of_neurons+1])
        for iter in range(self.no_of_iterations):
            for sample, true_value in zip(X, Y):
                sample = np.concatenate((sample, [1]))
                sample = np.array(sample, ndmin = 2).T
                out_array = np.zeros(self.output_nodes)
                for ind, val in enumerate(self.uni):
                    if val == true_value:
                        out_array[ind] = 1
                out_array = np.array(out_array, ndmin = 2).T

                # Forward Propagation

                for i in range(self.hidden_layers):
                    if i == 0:
                        z = np.dot(self.input_weights, sample)  
                    else:
                        z = np.dot(self.hidden_weights[:,:,i-1], np.array(biased_activation_values[:,i-1], ndmin = 2).T)
                    activation_values[:,i] = np.array(self.sigmoid(z), ndmin = 2).T
                    biased_activation_values[:,i] = np.concatenate((activation_values[:,i], [1]))
                final_value = self.softmax(np.dot(self.output_weights, np.array(biased_activation_values[:,-1], ndmin = 2).T))

                # Back Propagation
                
                final_error = true_value - final_value
                for i in range(self.hidden_layers):
                    if i == 0:
                        error = np.dot(self.output_weights.T, final_error)
                    else:
                        error = np.dot(self.hidden_weights[:,:,-i].T, np.array(error_matrix[:-1,-i], ndmin = 2).T)
                    error_matrix[:,-i-1][None] = np.array(error*np.array(self.sigmoid_derivative(biased_activation_values[:,-i-1]), ndmin = 2).T, ndmin = 2).T
                delta_output_weights += np.dot(final_error, np.array(biased_activation_values[:,-1], ndmin = 2))
                for i in range(self.hidden_layers-1):
                    delta_hidden_weights[:,:,-i-1] += np.dot(error_matrix[:-1,-i-1][None].T, np.array(biased_activation_values[:,-i-2], ndmin = 2))
                delta_input_weights += np.dot(error_matrix[:-1,0][None].T, sample.T)
            D_input = (1/len(Y))*delta_input_weights #+ 0.01*self.input_weights
            for i in range(self.hidden_layers-1):
                D_hidden[:,:,i] = (1/len(Y))*delta_hidden_weights[:,:,i] #+ 0.01*self.hidden_weights
            D_output = (1/len(Y))*delta_output_weights #+ 0.01*self.output_weights

            self.input_weights -= self.learning_rate*D_input
            for i in range(self.hidden_layers-1):
                self.hidden_weights[:,:,i] -= self.learning_rate*D_hidden[:,:,i]
            self.output_weights -= self.learning_rate*D_output
    
    def predict(self, X):
        activation_values = np.zeros([self.no_of_neurons, self.hidden_layers])
        biased_activation_values = np.zeros([self.no_of_neurons+1, self.hidden_layers])
        output = np.zeros(len(X))
        for index, sample in enumerate(X):
            sample = np.concatenate((sample, [1]))
            sample = np.array(sample, ndmin = 2).T
            
            # Forward Propagation
            
            for i in range(self.hidden_layers):
                if i == 0:
                    z = np.dot(self.input_weights, sample)  
                else:
                    z = np.dot(self.hidden_weights[:,:,i-1], np.array(biased_activation_values[:,i-1], ndmin = 2).T)
                activation_values[:,i] = np.array(self.sigmoid(z), ndmin = 2).T
                biased_activation_values[:,i] = np.concatenate((activation_values[:,i], [1]))
            final_value = self.softmax(np.dot(self.output_weights, np.array(biased_activation_values[:,-1], ndmin = 2).T))
            ind = np.argmax(final_value)
            output[index] = self.uni[ind]
        return output

    def accuracy(self, Y_pred, Y_true):
        
        # Computing accuracy of the predictions made!
        
        sum = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y_true[i]:
                sum += 1
        return (sum / len(Y_pred)) * 100        
    

