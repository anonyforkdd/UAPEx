# -*-encoding=utf-8-*-
################################################################################
#
# Copyright (c) 2021 xxx. All Rights Reserved
#
################################################################################
"""
explainer
Authors: xxx
Date:    2021/06/30

"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import inspect
import copy
import time
import random
from utils import StandardScaler
from utils import MinMaxScaler
from utils import index_to_adj_np
import utils
from model.vmrgae import VMR_GAE
import warnings
import heapq
import time
warnings.filterwarnings('ignore')
torch.set_default_tensor_type(torch.FloatTensor)
num_nodes = 263

def MC_sampling_rm_x_feature(related_nodeset, node_features, target_nodeset, endpoints, target_t, target_feature):
    # type: [other_nodes] -> S_A_flow:[timelen,[edge_index, edge_weight]]
    #                                   SG_A_flow:[timelen,[edge_index, edge_weight]]
    all_nodes = [i for i in range(num_nodes)]
    all_nodes.remove(endpoints[0])
    if endpoints[0]!=endpoints[1]:
        all_nodes.remove(endpoints[1])
    tmp_list = [i for i in range(related_nodeset.shape[0])]
    rand_num = random.choice(tmp_list)
    random.shuffle(tmp_list)
    selected_nodes = tmp_list[:rand_num]
    selected_nodeset = []
    for item in selected_nodes:
        selected_nodeset.append(related_nodeset[item])
        all_nodes.remove(related_nodeset[item])
    S_x = node_features.clone()
    S_x[target_t,all_nodes,target_feature] -= S_x[target_t,all_nodes,target_feature]
    SG_x = node_features.clone()
    for node_id in all_nodes:
        if node_id not in target_nodeset:
            SG_x[target_t,node_id,target_feature] -= SG_x[target_t,node_id,target_feature]
    return S_x, SG_x

def MC_sampling_rm_all_feature(remained_featureset, node_features, target_featureset, target_t, target_nodeset, feature_length):
    all_features = [i for i in range(feature_length)]
    tmp_list = [i for i in range(1, remained_featureset.shape[0]+1)]
    rand_num = random.choice(tmp_list)
    random.shuffle(tmp_list)
    selected_features = tmp_list[:rand_num]
    selected_featureset = []
    for item in selected_features:
        selected_featureset.append(remained_featureset[item-1])
        all_features.remove(remained_featureset[item-1])
    S_x = node_features.clone()
    for feature_id in all_features:
        S_x[target_t,target_nodeset,feature_id] -= S_x[target_t,target_nodeset,feature_id]
    SG_x = node_features.clone()
    for feature_id in all_features:
        if feature_id not in target_featureset:
            SG_x[target_t,target_nodeset,feature_id] -= SG_x[target_t,target_nodeset,feature_id]
    return S_x, SG_x

def MC_sampling_rm_from_history(related_nodeset, node_features, target_nodeset, endpoints, target_t):
    # type: [other_nodes] -> S_A_flow:[timelen,[edge_index, edge_weight]]
    #                                   SG_A_flow:[timelen,[edge_index, edge_weight]]
    tmp_list = [i for i in range(related_nodeset.shape[0])]
    rand_num = random.choice(tmp_list)
    random.shuffle(tmp_list)
    selected_nodes = tmp_list[:rand_num]
    selected_nodeset = []
    for item in selected_nodes:
        selected_nodeset.append(related_nodeset[item])
    all_nodes = [i for i in range(num_nodes)]
    all_nodes.remove(endpoints[0])
    if endpoints[0]!=endpoints[1]:
        all_nodes.remove(endpoints[1])
    S_nf = node_features[target_t][-1].clone()
    for node_id in all_nodes:
        if node_id not in selected_nodeset:
            S_nf[node_id] -= S_nf[node_id]
    SG_nf = node_features[target_t][-1].clone()
    for node_id in all_nodes:
        if (node_id not in selected_nodeset)&(node_id not in target_nodeset):
            SG_nf[node_id] -= SG_nf[node_id]
    return S_nf, SG_nf

def remove_node(A_flow_t, node_set): 
    # type: [edge_index, edge_weight], int -> [edge_index, edge_weight]
    need_removed_ind = []
    for i in range(len(node_set)):
        need_removed_ind.append(np.where((A_flow_t[0][:,0]==node_set[i]))[0])
        need_removed_ind.append(np.where((A_flow_t[0][:,1]==node_set[i]))[0])
    need_removed_ind = np.unique(np.concatenate(need_removed_ind))
    A_flow_t[0] = np.delete(A_flow_t[0], need_removed_ind, axis=0)
    A_flow_t[1] = np.delete(A_flow_t[1], need_removed_ind)
    return A_flow_t

def MC_sampling_rm_edge(related_nodeset, edge_index, edge_weight, target_nodeset, endpoints, node_graph_loc_list):
    # type: [other_nodes] -> S_A_flow:[timelen,[edge_index, edge_weight]]
    #                                   SG_A_flow:[timelen,[edge_index, edge_weight]]
    tmp_list = [i for i in range(related_nodeset.shape[0])]
    rand_num = random.choice(tmp_list)
    random.shuffle(tmp_list)
    selected_nodes = tmp_list[:rand_num]
    selected_nodeset = []
    for item in selected_nodes:
        selected_nodeset.append(related_nodeset[item])
    selected_nodeset.append(endpoints[0])
    if endpoints[0]!=endpoints[1]:
        selected_nodeset.append(endpoints[1])
    S_edge_index = edge_index.clone()
    S_edge_weight = edge_weight.clone()
    ind = np.unique(np.concatenate([node_graph_loc_list[i] for i in range(num_nodes) if i in selected_nodeset]))
    S_edge_index = S_edge_index[ind]
    S_edge_weight = S_edge_weight[ind]
    SG_edge_index = edge_index.clone()
    SG_edge_weight = edge_weight.clone()
    ind = np.unique(np.concatenate([node_graph_loc_list[i] for i in range(num_nodes) if (i in selected_nodeset)|(i in target_nodeset)]))
    SG_edge_index = SG_edge_index[ind]
    SG_edge_weight = SG_edge_weight[ind]
    return S_edge_index, S_edge_weight, SG_edge_index, SG_edge_weight

def get_L_hop_nodes(target_i,target_j,target_t,A_flow,target_nodeset,L=2):
    # type: int, [timelen,[edge_index, edge_weight]], [timeslots, target_nodes], int, int -> [timeslots, other_nodes]
    related_nodeset_t = []
    one_adj = index_to_adj_np(A_flow[target_t][0].T.cpu().numpy(), A_flow[target_t][1].cpu().numpy(), num_nodes).astype('bool')
    now_adj = one_adj.copy()
    arrivable_adj = one_adj.copy()
    for _ in range(L-1):
        now_adj = np.dot(now_adj, one_adj)
        arrivable_adj += now_adj
    for num_node in range(num_nodes):
        if (True in arrivable_adj[num_node, target_nodeset])|(True in arrivable_adj[target_nodeset, num_node]):
            if num_node not in target_nodeset:
                if (num_node!=target_i)&(num_node!=target_j):
                    related_nodeset_t.append(num_node)
    related_nodeset_t = np.array(related_nodeset_t)
    return related_nodeset_t

def get_reduced_target_graph(target_i, target_j, target_nodeset, A_flow_t, L=2):
    one_adj = index_to_adj_np(A_flow_t[0].T.cpu().numpy(), A_flow_t[1].cpu().numpy(), num_nodes)
    removed_nodeset = [i for i in range(num_nodes) if i not in target_nodeset]
    removed_nodeset.remove(target_i)
    if target_j in removed_nodeset:
        removed_nodeset.remove(target_j)
    one_adj[removed_nodeset] = 0.0
    one_adj[:,removed_nodeset] = 0.0
    one_adj = one_adj.astype('bool')
    target_nodeset = [target_i, target_j]
    related_nodeset_t = []
    now_adj = one_adj.copy()
    arrivable_adj = one_adj.copy()
    for _ in range(L-1):
        now_adj = np.dot(now_adj, one_adj)
        arrivable_adj += now_adj
    for num_node in range(num_nodes):
        if (True in arrivable_adj[num_node, target_nodeset])|(True in arrivable_adj[target_nodeset, num_node]):
            if num_node not in target_nodeset:
                related_nodeset_t.append(num_node)
    related_nodeset_t = np.array(related_nodeset_t)
    return np.array(related_nodeset_t)

def get_branchs(target_i, target_j, target_nodeset, A_flow_t):
    one_adj = index_to_adj_np(A_flow_t[0].T.cpu().numpy(), A_flow_t[1].cpu().numpy(), num_nodes)
    removed_nodeset = [i for i in range(num_nodes) if i not in target_nodeset]
    removed_nodeset.remove(target_i)
    if target_j in removed_nodeset:
        removed_nodeset.remove(target_j)
    one_adj[removed_nodeset] = 0.0
    one_adj[:,removed_nodeset] = 0.0
    adj_count = one_adj.astype('bool').astype('float32')
    degree = adj_count.sum(1) + adj_count.sum(0)
    degree[np.where(degree<=0)] = 2*num_nodes
    branchs_value = heapq.nsmallest(15, range(len(degree)), degree.take)
    branchs = []
    for i in range(len(branchs_value)):
        if len(np.where(target_nodeset==branchs_value[i])[0])!=0:
            branchs.append(np.where(target_nodeset==branchs_value[i])[0])
        if len(branchs)>=12:
            break
    return branchs

def identify_explain_region_ids(target_i, target_j, target_t, truth_value, all_h,x,A_flow,green_flow,
    A_scaler,truths, node_graph_loc_list,target_nodeset, min_regions=10, sample_times=100):
    #target_nodeset = np.delete(np.array([i for i in range(num_nodes)]),[target_i, target_j])
    target_nodeset = get_reduced_target_graph(target_i,target_j,target_nodeset,A_flow[target_t])
    edge_index, edge_weight = A_flow[-1][0], A_flow[-1][1]
    #edge_index = edge_index.T.cuda()
    while len(target_nodeset)>min_regions:
        best_branch = None
        best_branch_score = -10.0
        branchs = get_branchs(target_i, target_j, target_nodeset, A_flow[target_t])
        for branch in branchs:
            print("the number of rests:", len(target_nodeset), 'branch:', branch, 'removed node:', target_nodeset[branch])
            next_target_nodeset = target_nodeset.copy()
            next_target_nodeset = np.delete(target_nodeset, branch)
            related_nodeset = get_L_hop_nodes(target_i,target_j,target_t,A_flow,next_target_nodeset,L=2)
            if related_nodeset.shape[0]==0:
                print('score: None, no related other node, branch:', branch)
                continue
            m_S_G = []
            for _ in range(sample_times):
                S_edge_index, S_edge_weight, SG_edge_index, SG_edge_weight = MC_sampling_rm_edge(related_nodeset,
                    edge_index.T, edge_weight, next_target_nodeset, [target_i,target_j], node_graph_loc_list)
                #print(SG_edge_index.size(), SG_edge_weight.size(), S_edge_index.size(), S_edge_weight.size())
                S_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,x,
                    S_edge_index.T, S_edge_weight,green_flow,A_scaler,A_scaler.transform(truths))#.cpu().detach().numpy()[0]
                SG_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,x,
                    SG_edge_index.T, SG_edge_weight,green_flow,A_scaler,A_scaler.transform(truths))#.cpu().detach().numpy()[0]
                #print('Sample time', sample_time,':', SG_f_value, S_f_value)
                m_S_G.append(SG_f_value - S_f_value)
            m_S_G = (sum(m_S_G) / len(m_S_G)).cpu().detach().numpy()
            if m_S_G>best_branch_score:
                best_branch_score = m_S_G
                best_branch = branch
                print('New best score:', m_S_G, 'branch:', best_branch)
            else:
                print('score:', m_S_G, 'is not greater than the best, branch:', branch)
        target_nodeset = np.delete(target_nodeset, best_branch)
        print(target_nodeset)
        target_nodeset = get_reduced_target_graph(target_i,target_j,target_nodeset,A_flow[target_t])
        print(target_nodeset)
    return target_nodeset

def identify_explain_historical_ids(target_i, target_j, target_t, truth_value, all_h,x,A_flow,green_flow,
    A_scaler,truths, node_graph_loc_list,target_nodeset, min_regions=10, sample_times=100):
    #target_nodeset = np.delete(np.array([i for i in range(213)]),[target_i, target_j])
    target_nodeset = get_reduced_target_graph(target_i,target_j,target_nodeset,A_flow[target_t])
    edge_index, edge_weight = A_flow[-1][0], A_flow[-1][1]
    while len(target_nodeset)>min_regions:
        best_branch = None
        best_branch_score = -10.0
        branchs = get_branchs(target_i, target_j, target_nodeset, A_flow[target_t])
        for branch in branchs:
            print("the number of rests:", len(target_nodeset), 'branch:', branch, 'removed node:', target_nodeset[branch])
            next_target_nodeset = target_nodeset.copy()
            next_target_nodeset = np.delete(target_nodeset, branch)
            related_nodeset = get_L_hop_nodes(target_i,target_j,target_t,A_flow,next_target_nodeset,L=2)
            if related_nodeset.shape[0]==0:
                print('score: None, no related other node, branch:', branch)
                continue
            m_S_G = []
            for _ in range(sample_times):
                S_h, SG_h = MC_sampling_rm_from_history(related_nodeset, all_h ,next_target_nodeset, [target_i,target_j], target_t)
                S_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,S_h,x,
                    edge_index, edge_weight,green_flow,A_scaler,A_scaler.transform(truths)).cpu().detach().numpy()[0]
                SG_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,SG_h,x,
                    edge_index, edge_weight,green_flow,A_scaler,A_scaler.transform(truths)).cpu().detach().numpy()[0]
                m_S_G.append(SG_f_value - S_f_value)
            m_S_G = sum(m_S_G) / len(m_S_G)
            if m_S_G>best_branch_score:
                best_branch_score = m_S_G
                best_branch = branch
                print('New best score:', m_S_G, 'branch:', best_branch)
            else:
                print('score:', m_S_G, 'is not greater than the best, branch:', branch)
        target_nodeset = np.delete(target_nodeset, best_branch)
    return target_nodeset


def identify_feature_importance_score(target_nodeset, target_i, target_j, target_t,
    truth_value, all_h, x, d, A_flow, sample_times=150, feature_length=19):
    target_featureset = np.array([i for i in range(feature_length)])
    feature_important_score = [feature_length-1 for i in range(feature_length)]
    edge_index, edge_weight = torch.tensor(A_flow[-1][0]).cuda(), torch.tensor(A_flow[-1][1]).cuda()
    for times in range(feature_length-1):
        best_branch = None
        best_branch_score = -10.0
        branchs = np.array([i for i in range(len(target_featureset))])
        np.random.shuffle(branchs)
        for branch in branchs:
            next_target_featureset = target_featureset.copy()
            next_target_featureset = np.delete(target_featureset, branch)
            remained_featureset = np.array(list(set([i for i in range(feature_length)])-set(next_target_featureset)))
            print("the number of rests:", len(target_featureset), 'branch:', branch, 'removed node:',target_t, target_featureset[branch])
            if remained_featureset.shape[0]==0:
                print('Cause ERROR')
                continue
            m_S_G = []
            for _ in range(sample_times):
                S_x, SG_x = MC_sampling_rm_all_feature(remained_featureset, x, next_target_featureset, target_t, target_nodeset, feature_length)
                S_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,S_x,d,
                    edge_index, edge_weight,A_scaler.transform(truth_value)).cpu().detach().numpy()[0]
                SG_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,SG_x,d,
                    edge_index, edge_weight,A_scaler.transform(truth_value)).cpu().detach().numpy()[0]
                m_S_G.append(SG_f_value - S_f_value)
            m_S_G = sum(m_S_G) / len(m_S_G)
            if m_S_G>best_branch_score:
                best_branch_score = m_S_G
                best_branch = branch
                print('New best score:', m_S_G, 'branch:', best_branch)
            else:
                print('score:', m_S_G, 'is not greater than the best, branch:', branch)
        feature_important_score[target_featureset[best_branch]] = times
        target_featureset = np.delete(target_featureset, best_branch)
    return feature_important_score

def identify_regional_feature_importance(target_nodeset_origin,target_i, target_j, target_t,
    truth_value, all_h, x, d, A_flow, sample_times=50, feature_length=19, branch_sample_times=30):
    sort_importance_lists = []
    edge_index, edge_weight = torch.tensor(A_flow[-1][0]).cuda(), torch.tensor(A_flow[-1][1]).cuda()
    for target_feature in range(feature_length):
        sort_importance_list = []
        target_nodeset = target_nodeset_origin.copy()
        for _ in range(9):
            best_branch = None
            best_branch_score = -10.0
            for branch in [i for i in range(len(target_nodeset))]:
                print("the number of rests:", len(target_nodeset), 'branch:', branch, 'removed node:',target_t, target_nodeset[branch])
                next_target_nodeset = target_nodeset.copy()
                next_target_nodeset = np.delete(target_nodeset, branch)
                related_nodeset = get_L_hop_nodes(target_i,target_j,target_t,A_flow,next_target_nodeset,L=2)
                if related_nodeset.shape[0]==0:
                    print('score: None, no related other node, branch:', branch)
                    continue
                m_S_G = []
                for _ in range(sample_times):
                    S_x, SG_x = MC_sampling_rm_x_feature(related_nodeset, x ,next_target_nodeset, [target_i,target_j], target_t, target_feature)
                    S_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,S_x,d,edge_index,
                        edge_weight,A_scaler.transform(truth_value)).cpu().detach().numpy()[0]
                    SG_f_value = gnn.edge_pred_contribution(target_i,target_j,target_t,all_h,SG_x,d,edge_index,
                        edge_weight,A_scaler.transform(truth_value)).cpu().detach().numpy()[0]
                    m_S_G.append(SG_f_value - S_f_value)
                m_S_G = sum(m_S_G) / len(m_S_G)
                if m_S_G>best_branch_score:
                    best_branch_score = m_S_G
                    best_branch = branch
                    print('New best score:', m_S_G, 'branch:', best_branch)
                else:
                    print('score:', m_S_G, 'is not greater than the best, branch:', branch)
            sort_importance_list.append(target_nodeset[best_branch])
            target_nodeset = np.delete(target_nodeset, best_branch)
        sort_importance_list.append(target_nodeset[0])
        sort_importance_lists.append(np.array(sort_importance_list))
    sort_importance_lists = np.array(sort_importance_lists)
    return sort_importance_lists

def node_graph_loc_query(A_flow_t):
    the_index = A_flow_t[0].T.cpu().detach().numpy()
    node_graph_loc_list = []
    for node_id in range(num_nodes):
        one_node_graph_loc = np.concatenate([np.where((the_index[:,0]==node_id))[0],np.where((the_index[:,1]==node_id))[0]])
        one_node_graph_loc = np.unique(one_node_graph_loc)
        node_graph_loc_list.append(one_node_graph_loc)
    return node_graph_loc_list
        


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='[UAPEx] Model Explainer')
    parser.add_argument('--starttime', type=int, default=160, help='starttime')
    parser.add_argument('--endtime', type=int, default=192, help='endtime')
    parser.add_argument('--num_nodes', type=int, default=263, help='The number of nodes in the graph')
    parser.add_argument('--timelen', type=int, default=3, help='The length of input sequence')
    parser.add_argument('--hidden_dim', type=int, default=32, help='The dimensionality of the hidden state')
    parser.add_argument('--rnn_layer', type=int, default=2, help='The number of RNN layers')
    parser.add_argument('--delay', type=int, default=0, help='delay to apply kld_loss')
    parser.add_argument('--clip_max_value', type=int, default=1, help='clip the max value')
    parser.add_argument('--align', type=bool, default=True,
                        help='Whether or not align the distributions of two modals')
    parser.add_argument('--x_feature', type=bool, default=False,
                        help='X is a feature matrix (if True) or an identity matrix (otherwise)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Data path')
    parser.add_argument('--checkpoints', type=str, default='./nyc/checkpoints', help='Checkpoints path')
    parser.add_argument('--result_path', type=str, default='./nyc/results', help='result path')

    #Explainer parameters
    #There is NO node features in NYC! task must less than 2!
    parser.add_argument('--task', type=int, default=0, help='0:regional, 1:historcal, 2:feature, 3:each feature')
    parser.add_argument('--sample_times', type=int, default=50, help='Sample times')
    parser.add_argument('--target_i', type=int, default=120, help='the O of target OD')
    parser.add_argument('--target_j', type=int, default=164, help='the D of target OD')
    parser.add_argument('--target_t', type=int, default=-1, help='target time')
    parser.add_argument('--min_regions', type=int, default=10, help='min number of regions')
    parser.add_argument('--feature_length', type=int, default=263, help='feature length')

    args = parser.parse_args()
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
    device = torch.device('cuda:0')
    num_nodes = args.num_nodes

    #read data from file
    all_len = 84
    true_dis = np.load('%sdismatrix.npy' %(args.data_path))
    A_flow = np.load('%strain_data.npy' %(args.data_path), allow_pickle=True)[args.starttime-(args.endtime-all_len):args.timelen+args.starttime-(args.endtime-all_len)]
    val_data = np.load('%sval_data.npy' %args.data_path, allow_pickle=True)[args.timelen+args.starttime-(args.endtime-all_len)-1]
    test_data = np.load('%stest_data.npy' %args.data_path, allow_pickle=True)[args.timelen+args.starttime-(args.endtime-all_len)-1]
    green_flow = np.load('%sgreen_data.npy' %(args.data_path), allow_pickle=True)[args.starttime-(args.endtime-all_len):args.timelen+args.starttime-(args.endtime-all_len)]

    #data scaler
    new_A_flow = []
    truths = []
    for i in range(len(A_flow)):
        new_A_flow.append([np.array(A_flow[i][0]).astype("int"), np.array(A_flow[i][1]).astype("float32")])
        truths.append(utils.index_to_adj_np(new_A_flow[i][0],new_A_flow[i][1], args.num_nodes))
    truths = np.stack(truths, axis=0)
    if args.clip_max_value==1:
        max_value = 50
    else:
        print(np.concatenate(A_flow[:,1]).max())
        max_value = np.concatenate(A_flow[:,1]).max()
    A_scaler = MinMaxScaler(0, max_value)
    A_flow = new_A_flow
    for i in range(args.timelen):
        A_flow[i][1] = A_scaler.transform(A_flow[i][1])

    for i in range(len(green_flow)):
        green_flow[i][0] = np.array(green_flow[i][0]).astype("int")
        green_flow[i][1] = np.array(green_flow[i][1]).astype("float32")
    yellow_scaler = MinMaxScaler(0, np.concatenate(green_flow[:,1]).max())
    for i in range(args.timelen):
        green_flow[i][1] = yellow_scaler.transform(green_flow[i][1])

    #load into torch
    mask = np.zeros((args.num_nodes, args.num_nodes))
    for i in range(args.timelen):
        the_adj = utils.index_to_adj_np(A_flow[i][0],A_flow[i][1], args.num_nodes)
        mask[np.where(the_adj>(2/max_value))] = 1.0
    mask = torch.tensor(mask, dtype=torch.bool).to(device)
    x = torch.eye(args.num_nodes).to(device)
    truths = torch.from_numpy(truths).to(device)
    print(truths.max())
    for i in range(args.timelen):
        green_flow[i][0] = torch.tensor(green_flow[i][0]).T.to(device)
        green_flow[i][1] = torch.tensor(green_flow[i][1]).to(device)
        A_flow[i][0] = torch.tensor(A_flow[i][0]).T.to(device)
        A_flow[i][1] = torch.tensor(A_flow[i][1]).to(device)
    
    gnn = VMR_GAE(x_dim=x.shape[-1], h_dim=args.hidden_dim,
                num_nodes=args.num_nodes, n_layers=args.rnn_layer, device=device,
                eps=1e-10, align=args.align, is_region_feature=args.x_feature)
    
    if not os.path.isfile('%s/model.pth' % args.checkpoints):
        print('Checkpoint does not exist.')
        exit()
    else:
        gnn.load_state_dict(torch.load('%s/model.pth' % args.checkpoints).state_dict())
    gnn = gnn.to(device)
    kld_loss_tvge, kld_loss_avde, pis_loss, all_h, all_enc_mean, all_prior_mean, all_enc_d_mean,\
            all_dec_t, all_z_in, all_z_out = gnn(x, A_flow, green_flow, mask,A_scaler, truths)
    node_graph_loc_list = node_graph_loc_query(A_flow[args.target_t])

    target_i = args.target_i
    target_j = args.target_j
    truth_value = truths[args.target_t][args.target_i][args.target_j]
    print('This is',i,'epoch',target_i,target_j,truth_value)
    start_region_neighbors = heapq.nlargest(35, range(len(true_dis[target_i])), true_dis[target_i].take)
    end_region_neighbors = heapq.nlargest(35, range(len(true_dis[target_j])), true_dis[target_j].take)
    target_nodeset = list(set(start_region_neighbors+end_region_neighbors))
    target_nodeset.remove(target_i)
    if target_i!=target_j:
        target_nodeset.remove(target_j)
    target_nodeset = np.array(target_nodeset)   
    

    #---------------Main functions----------------
    
    if args.task==0:
        #reduce the region number to explore which regions are important
        startt=time.time()
        explain_regions = identify_explain_region_ids(args.target_i,args.target_j,args.target_t,
                                                      truth_value,all_h,x,A_flow,green_flow,A_scaler,truths,
                                                      node_graph_loc_list, target_nodeset)
        np.save('%sexplain_regions_%d_%d_%d.npy' %(args.result_path,target_i,target_j,args.starttime), explain_regions)
        endt=time.time()
        print('result:', explain_regions, 'O:', target_i, 'D:', target_j)
        print('identify_explain_region_ids: %s Seconds'%(endt-startt))

    if args.task==1:
        #reduce the region number according to $h$ to explore which historical regions are important
        startt=time.time()
        explain_historical_regions = identify_explain_historical_ids(args.target_i,args.target_j,args.target_t,
                                                      truth_value,all_h,x,A_flow,green_flow,A_scaler,truths,
                                                      node_graph_loc_list, target_nodeset)
        np.save('%sexplain_historical_regions_%d_%d_%d.npy' %(args.result_path,target_i,target_j,args.starttime), explain_historical_regions)
        endt=time.time()
        print('result:', explain_historical_regions, 'O:', target_i, 'D:', target_j)
        print('identify_explain_historical_ids: %s Seconds'%(endt-startt))

    if args.task>=2:
        #calculate the importance score among features
        startt=time.time()
        explain_regions = np.load('%s/explain_regions_%d_%d_%d.npy' %(args.data_path,target_i,target_j,args.starttime))
        feature_important_score = identify_feature_importance_score(explain_regions, args.target_i, args.target_j,
                                                                    args.target_t, args.truth_value, all_h, x, d, A_flow)
        np.save('%sfeature_important_score_%d_%d_%d.npy' %(args.result_path,target_i,target_j,args.starttime), feature_important_score)
        endt=time.time()
        print('identify_feature_importance_score: %s Seconds'%(endt-startt))
 
        if args.task==3:
            #sort the reduced regions for each feature dim
            startt=time.time()
            sort_importance_lists = identify_regional_feature_importance(explain_regions,target_i, target_j,
                                                                     args.target_t,args.truth_value, all_h,
                                                                     x, d, A_flow)
            np.save('%ssort_importance_lists_%d_%d_%d.npy' %(args.result_path,target_i,target_j,args.starttime), sort_importance_lists)
            endt=time.time()
            print('sort_importance_lists: %s Seconds'%(endt-startt))

