# -*-encoding=utf-8-*-
################################################################################
#
# Copyright (c) 2021 xxx, Inc. All Rights Reserved
#
################################################################################
"""
utils of model
Authors: xxx
Date:    2021/04/26
"""
import numpy as np
import torch
import torch.nn as nn
import math

def validate(preds, test_index, test_value, flag='val'):
    MAE = 0
    RMSE = 0
    MAPE = 0
    count = 0
    #print(test_index[0][0])
    for i in range(len(test_index)):
        #print(test_value[i])
        if test_value[i]>2:
            MAE += abs(test_value[i] - preds[int(test_index[i][0])][int(test_index[i][1])])
            RMSE += abs(test_value[i] - preds[int(test_index[i][0])][int(test_index[i][1])]) ** 2
            MAPE += abs(test_value[i] - preds[int(test_index[i][0])][int(test_index[i][1])]) / test_value[i]
            count += 1
    MAE = MAE / count
    RMSE = math.sqrt(RMSE / count)
    MAPE = MAPE / count
    #print(flag, "MAE:", MAE, 'RMSE:', RMSE, 'MAPE:', MAPE)
    return MAE, RMSE, MAPE

def valid5(preds, test_index, test_value, flag='val'):
    MAE = 0
    RMSE = 0
    MAPE = 0
    count = 0
    for i in range(len(test_index)):
        if test_value[i]>5:
            MAE += abs(test_value[i] - preds[test_index[i][0]][test_index[i][1]])
            RMSE += abs(test_value[i] - preds[test_index[i][0]][test_index[i][1]]) ** 2
            MAPE += abs(test_value[i] - preds[test_index[i][0]][test_index[i][1]]) / test_value[i]
            count += 1
    MAE = MAE / count
    RMSE = math.sqrt(RMSE / count)
    MAPE = MAPE / count
    print(flag, "MAE:", MAE, 'RMSE:', RMSE, 'MAPE:', MAPE)

def index_to_adj_np(edge_index, edge_weight, num_nodes):
    #print(edge_index.shape, edge_weight.shape)
    adj = np.zeros((num_nodes, num_nodes))
    for i in range(len(edge_weight)):
        adj[edge_index[i][0]][edge_index[i][1]] += edge_weight[i]
    return adj

def cal_mask(init_mask, masked_edge_index):
    '''
    (numpy.bool, list, int) -> numpy.bool
    '''
    mask = init_mask.copy()
    for i in range(len(masked_edge_index)):
        mask[masked_edge_index[i][0]][masked_edge_index[i][1]] = False
    return mask

def masked_pisloss(loss_func, inputs, edge_weight, edge_index, zero_edge_index):
    '''
    (torch.nn.loss, tensor, tensor, tensor, tensor) -> tensor
    '''
    loss_1 = loss_func(inputs)

def arctanh_np(inputs, eps=1e-6):
    return 0.5 * np.log((1+inputs)/(1-inputs+eps))

def arctanh(inputs, eps=1e-6):
    eps = torch.tensor(eps)
    return 0.5 * torch.log((1+inputs)/(1-inputs+eps))

def calculate_normalized_laplacian(adj):
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    normalized_laplacian = (adj+np.eye(adj.shape[0])).dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return normalized_laplacian

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        print('mean:', self.mean, 'std:', self.std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class MinMaxScaler:
    """
    MinMax the input
    """

    def __init__(self, minvalue, maxvalue):
        self.minvalue = minvalue
        self.maxvalue = maxvalue
        print('min:', self.minvalue, 'max:', self.maxvalue)

    def transform(self, data):
        return (data - self.minvalue) / (self.maxvalue - self.minvalue)

    def inverse_transform(self, data):
        return (data * (self.maxvalue - self.minvalue)) + self.minvalue
