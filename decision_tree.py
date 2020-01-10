#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
class decision_tree():
    
    def __init__(self, max_depth = 10, max_size = 5):
        self.max_depth = max_depth
        self.max_size = max_size
        self.pred = list()
        
    def label_freq(self, group):
        freq = Counter(group).most_common()
        return freq
    
    def gini_index(self, groups):
        gini = 0
        for group in groups:
            freq = self.label_freq(group)
            gini += sum([[freq[label][1]/len(groups)][0]**2 for label in range(len(freq))])
        return 1 - gini
    
    def information_gain(self, parent_node, l_child, r_child):
        weighted_gini = self.gini_index(l_child)*(len(l_child)/(len(l_child) + len(r_child))) + self.gini_index(r_child)*(len(r_child)/(len(l_child) + len(r_child)))
        return self.gini_index(parent_node) - weighted_gini
    
    def split(self, node):
        best_condition = 0
        max_gain = 0
        best_split_feature = 0
        node_val = 0
        index = 0
        l_node = 0
        r_node = 0
        np.random.seed(3)
        for ind in range(len(node[0])-1):
                condition = np.random.choice(node[0])
                l_child, r_child = self.make_split(node, condition, ind)
                if len(l_child) != 0 and len(r_child) != 0:
                    gain = self.information_gain(node, l_child, r_child)
                    if gain > max_gain:
                        max_gain = gain
                        best_condition = condition
                        node_val = node[ind]
                        index = ind
                        l_node = l_child
                        r_node = r_child
        return {'value': node_val, 'ind': index, 'l': l_node, 'r': r_node}
    
    def leaf_node(self, node):
        labels = [sample[-1] for sample in node]
        most_common_label = Counter(labels).most_common()
        return most_common_label[0][0]
    
    def make_split(self, data, condition, feature):
        l_child = []
        r_child = []
        for sample in data:
            if self.test_split(sample, condition, feature):
                l_child.append(sample)
            else:
                r_child.append(sample)
        return l_child, r_child
    
    def test_split(self, data, condition, feature):
        value = data[feature]
        return value >= condition
    
    def make_tree(self, train, test):
        
        ''' It takes in your training and testing data and returns the values of testing data in a tree like format
         but user might not be able to understand it if a lot of data is given!'''
        if type(train) != np.ndarray:
            train = train.to_numpy()
        if type(test) != np.ndarray:
            test = test.to_numpy()
        root_node = self.split(train)
        self.grow_tree(root_node, 1)
        for val in test:
            pre = self.show_tree(root_node, val)
            self.pred.append(pre)
        return self.pred
    
    def prediction_labels(self, values):
        labels =list()
        for arr in values[0]:
            label = arr[-1]
            labels.append(label)
        return labels
    
    def show_tree(self, node, value):
        if value[node['ind']] < node['value'][node['ind']]:
            if (type(node['l'])) == dict:
                return self.show_tree(node['l'], value)
            else:
                return node['l']
        else:
            if type(node['r']) == dict:
                return self.show_tree(node['r'], value)
            else:
                return node['r']
            
    def grow_tree(self, node, depth):
        l_node = node['l']
        r_node = node['r']
        if not l_node or not r_node:
            if not l_node and not r_node:
                return
            l_node = self.leaf_node(l_node + r_node)
            r_node = self.leaf_node(r_node + l_node)
            return
        if depth == self.max_depth:
            l_node = self.leaf_node(l_node)
            r_node = self.leaf_node(r_node)
            return None
        if len(l_node) <= self.max_size:
            l_node = self.leaf_node(l_node)
        else:
            node['l_node'] = self.split(l_node)
            self.grow_tree(node['l_node'], depth + 1)
        if len(r_node) <= self.max_size:
            r_node = self.leaf_node(r_node)
        else:
            node['r_node'] = self.split(r_node)
            self.grow_tree(node['r_node'], depth + 1)
            
    def accuracy(self, pred_label, true_label):
        ''' It returns the accuracy of prediction!
            It takes predicted value and true output labels as parameters!'''
        sum = 0
        for val in self.pred:
            for label in true_label:
                count = 0
                for i in range(val[0].shape[0]):
                    if val[0][i] ==label[i]:
                        count += 1
                if count == val[0].shape[0]:
                    sum += 1
        return (sum/len(pred_label))*100

