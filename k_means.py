# -*- coding: utf-8 -*-
"""K Means.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vjmul6KVWfsTZOIa2yMRVFoRWKNsDl4u
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class k_means():
    
     ''' Just call the predict() method while giving providing the dataset for clustering'''
        
    def __init__(self, k = 3, no_of_iteration = 1000, random_state = 10):
        self.k = k
        self.no_of_iteration = no_of_iteration
        self.random_state = random_state
        self.clusters = [[] for i in range(self.k)]
        self.dynamic_centroid = []
    
    def calculate_distance(self, centroid, point):
        return np.sqrt(np.sum((centroid - point)**2))
    
    def nearest_centroid(self, point, centroids):
        distance = [self.calculate_distance(centroid, point) for centroid in centroids]
        
        nearest_centroid_index = np.argmin(distance)
        return nearest_centroid_index
    
    def make_clusters(self, centroid):
        cluster = [[] for i in range(self.k)]
        for index, point in enumerate(self.X):
            centroid_index = self.nearest_centroid(point, centroid)
            cluster[centroid_index].append(index)
        return cluster
    
    def compute_centroid(self, clusters):
        centroid = np.zeros([self.k, self.no_of_features])
        for index, single_cluster in enumerate(clusters):
            centroid[index] = np.mean(self.X[single_cluster], axis = 0)
        return centroid
    
    def check_convergence(self, previous_centroid, latest_centroid):
        total_distance = [self.calculate_distance(previous_centroid[i], latest_centroid[i]) for i in range(self.k)]
        return sum(total_distance) == 0
    
    def label(self, clusters):
        label = np.zeros(self.no_of_samples)
        for index, cluster in enumerate(clusters):
            for sample_index in cluster:
                label[sample_index] = index
        return label
    
    def predict(self, X):
        self.X = X
        self.no_of_samples, self.no_of_features = X.shape

        #  Randomly initializing the centroid!

        np.random.seed(self.random_state)
        random_index = np.random.choice(self.no_of_samples, self.k, replace = False)
        self.dynamic_centroid = [self.X[index] for index in random_index]
        for i in range(self.no_of_iteration):
           
            # Updating Clusters!
            
            self.clusters = self.make_clusters(self.dynamic_centroid)

            # Updating Centroids!

            prev_centroid = self.dynamic_centroid
            self.dynamic_centroid = self.compute_centroid(self.clusters)

            if self.check_convergence(prev_centroid, self.dynamic_centroid):
                string = str(i+1)
                print("Convergence occured at", string+'th', "iteration!")
                break
        self.labels = self.label(self.clusters)
        return self.labels
    
    def plot(self):
        
        ''' If 2 features are given then clusters can be visualized with this function! '''

        figure, ax = plt.subplots(figsize = (12,8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)   
        for point in self.dynamic_centroid:
            ax.scatter(*point, marker = 'X', color = "black", linewidth = 2)
        plt.show()
