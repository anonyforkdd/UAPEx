# -*-encoding=utf-8-*-
################################################################################
#
# Copyright (c) 2021 xxx, Inc. All Rights Reserved
#
################################################################################
"""
model
Authors: xxx
Date:    2021/04/13
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import copy
import time
from model.dgcn import GCNConv
from utils import calculate_normalized_laplacian
from utils import StandardScaler
from utils import MinMaxScaler
import utils

torch.set_default_tensor_type(torch.FloatTensor)

class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer
        
        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(self.n_layer):
            if i==0:
                self.weight_xz.append(GCNConv(input_size, hidden_size))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size))
                self.weight_xr.append(GCNConv(input_size, hidden_size))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size))
                self.weight_xh.append(GCNConv(input_size, hidden_size))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size))
        self.layer_module = nn.ModuleList(self.weight_xz+self.weight_hz+self.weight_xr+self.weight_hr+self.weight_xh+self.weight_hh)
    
    def forward(self, inp, edge_index, edge_weight, h):
        h_out = []
        for i in range(self.n_layer):
            if i==0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edge_index, edge_weight) + self.weight_hz[i](h[i], edge_index, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edge_index, edge_weight) + self.weight_hr[i](h[i], edge_index, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edge_index, edge_weight) + self.weight_hh[i](r_g * h[i], edge_index, edge_weight))
                h_out.append(z_g * h[i] + (1 - z_g) * h_tilde_g)
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i-1], edge_index, edge_weight) + self.weight_hz[i](h[i], edge_index, edge_weight))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i-1], edge_index, edge_weight) + self.weight_hr[i](h[i], edge_index, edge_weight))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i-1], edge_index, edge_weight) + self.weight_hh[i](r_g * h[i], edge_index, edge_weight))
                h_out.append(z_g * h[i] + (1 - z_g) * h_tilde_g)
        return h_out

# main framework
class VMR_GAE(nn.Module):
    def __init__(self, x_dim, h_dim, num_nodes, n_layers, device,
                    eps=1e-10, align=True, is_region_feature=True, conv='DGCN', bias=False):
        super(VMR_GAE, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.e_x_dim = h_dim // 2
        self.e_d_dim = h_dim // 2
        self.n_layers = n_layers
        self.eps = eps
        self.num_nodes = num_nodes
        self.align = align
        self.is_region_feature = is_region_feature
        self.device = device

        if conv == 'DGCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_e_x = nn.Sequential(nn.Linear(self.e_x_dim, h_dim), nn.ReLU())
            self.phi_d = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())

            self.phi_z_out = nn.Sequential(
                nn.Linear(self.e_x_dim + self.e_d_dim, 4*(self.e_x_dim + self.e_d_dim)),
                nn.PReLU(),
                nn.Linear(4*(self.e_x_dim + self.e_d_dim), 64),
                nn.PReLU())
            self.phi_z_in = nn.Sequential(
                nn.Linear(self.e_x_dim + self.e_d_dim, 4*(self.e_x_dim + self.e_d_dim)),
                nn.PReLU(),
                nn.Linear(4*(self.e_x_dim + self.e_d_dim), 64),
                nn.PReLU())
            self.phi_dec = nn.Sequential(
                nn.Linear(128, 256),
                nn.PReLU(),
                nn.Linear(256, 256),
                nn.PReLU(),
                nn.Linear(256, 1))
            
            self.enc = GCNConv(h_dim + h_dim, h_dim).to(self.device)
            self.enc_mean = GCNConv(h_dim, self.e_x_dim).to(self.device)
            self.enc_std = GCNConv(h_dim, self.e_x_dim).to(self.device)

            self.sup_enc = GCNConv(h_dim, h_dim).to(self.device)
            self.sup_enc_mean = GCNConv(h_dim, self.e_x_dim).to(self.device)
            self.sup_enc_std = GCNConv(h_dim, self.e_x_dim).to(self.device)
            
            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, self.e_x_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, self.e_x_dim), nn.Softplus())
            
            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)

        if not align:
            self.d_prior_mean = nn.Parameter(torch.zeros(num_nodes, self.e_d_dim), requires_grad=False)
            self.d_prior_std = nn.Parameter(torch.ones(num_nodes, self.e_d_dim), requires_grad=False)
        self.pisloss = nn.PoissonNLLLoss(log_input=False, full=True, reduction='none')
        self.mseloss = nn.MSELoss(reduction='none')
        

    def forward(self, x, A_flow, supple_flow, mask, A_scaler, truths, hidden_in=None):

        kld_loss_tvge = 0
        kld_loss_avde = 0
        pis_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_enc_d_mean, all_enc_d_std = [], []
        all_z_in, all_z_out = [], []
        all_dec_t = []
        all_h = []

        # the main process
        if hidden_in is None:
            h = torch.zeros(self.n_layers, self.num_nodes, self.h_dim, device = self.device, requires_grad=True)
        else:
            h = torch.tensor(hidden_in, device = self.device, requires_grad=True)
        all_h.append(h)

        for t in range(len(A_flow)):
            #print("Calculate time:", t)
            if self.is_region_feature:
                phi_x_t = self.phi_x(x[t])
                phi_d_t = self.phi_d(x[t])
            else:
                phi_x_t = self.phi_x(x)
                phi_d_t = self.phi_d(x)
            

            edge_index, edge_weight = A_flow[t][0], A_flow[t][1]
            edge_index_sup, edge_weight_sup = supple_flow[t][0], supple_flow[t][1]
            #encoder of temporal variational graph encoder
            enc_t = F.relu(self.enc(torch.cat([phi_x_t, all_h[t][-1]], 1), edge_index, edge_weight))
            enc_mean_t = self.enc_mean(enc_t, edge_index, edge_weight)
            enc_std_t = F.softplus(self.enc_std(enc_t, edge_index, edge_weight))

            #encoder of supplement
            enc_d_t = F.relu(self.sup_enc(phi_d_t, edge_index_sup, edge_weight_sup))
            enc_d_mean_t = self.sup_enc_mean(enc_d_t, edge_index_sup, edge_weight_sup)
            enc_d_std_t = F.softplus(self.sup_enc_std(enc_d_t, edge_index_sup, edge_weight_sup))


            #prior
            prior_t = self.prior(all_h[t][-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            #sampling and reparameterization
            e_x_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_e_x_t = self.phi_e_x(e_x_t)
            e_d_t = self._reparameterized_sample(enc_d_mean_t, enc_d_std_t)
            
            #decoder
            z_t_in = torch.unsqueeze(self.phi_z_in(torch.cat([e_x_t, e_d_t], 1)), 0)
            z_t_out = torch.unsqueeze(self.phi_z_out(torch.cat([e_x_t, e_d_t], 1)), 1)
            z_t_in = torch.cat([z_t_in for i in range(self.num_nodes)], dim=0)
            z_t_out = torch.cat([z_t_out for i in range(self.num_nodes)], dim=1)
            z_t = torch.cat([z_t_out, z_t_in], dim=2).view((self.num_nodes*self.num_nodes, -1))
            dec_t = F.sigmoid(self.phi_dec(z_t)).view((self.num_nodes, self.num_nodes))
            
            

            #recurrence
            h_t = self.rnn(torch.cat([phi_x_t, phi_e_x_t], 1), edge_index, edge_weight, all_h[t])
            all_h.append(h_t)

            kld_loss_tvge += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)
            if self.align:
                kld_loss_avde += 0.2 * self._kld_gauss(enc_d_mean_t, enc_d_std_t, prior_mean_t.data, prior_std_t.data)
            else:
                kld_loss_avde += 0.2 * self._kld_gauss(enc_d_mean_t, enc_d_std_t, self.d_prior_mean, self.d_prior_std)

            if (len(A_flow)-t)<=24:
                pis_loss += self.masked_pisloss(dec_t, truths[t], mask, A_scaler)
            #sim_loss += self.Regularization_loss(z_t_in, poi_lambda_in[t], z_t_out)

            all_enc_std.append(enc_std_t)
            all_enc_mean.append(enc_mean_t)
            all_enc_d_std.append(enc_d_std_t)
            all_enc_d_mean.append(enc_d_mean_t)
            all_prior_mean.append(prior_mean_t)
            all_prior_std.append(prior_std_t)
            all_dec_t.append(dec_t)
            all_z_in.append(z_t_in)
            all_z_out.append(z_t_out)

        return kld_loss_tvge, kld_loss_avde, pis_loss, all_h, all_enc_mean, all_prior_mean, all_enc_d_mean, all_dec_t, all_z_in, all_z_out

    def edge_pred_contribution(self,target_i,target_j,target_t,all_h,x,edge_index, edge_weight,supple_flow,A_scaler,truths,iota=4.605):
        dec_t = 0
        for t in range(target_t, target_t+1):
            if self.is_region_feature:
                phi_x_t = self.phi_x(x[t])
                phi_d_t = self.phi_d(x[t])
            else:
                phi_x_t = self.phi_x(x)
                phi_d_t = self.phi_d(x)
            
            edge_index_sup, edge_weight_sup = supple_flow[t][0], supple_flow[t][1]
            #encoder of temporal variational graph encoder
            if torch.is_tensor(all_h):
                enc_t = F.relu(self.enc(torch.cat([phi_x_t, all_h], 1), edge_index, edge_weight))
            else:
                enc_t = F.relu(self.enc(torch.cat([phi_x_t, all_h[t][-1]], 1), edge_index, edge_weight))
            enc_mean_t = self.enc_mean(enc_t, edge_index, edge_weight)
            enc_std_t = F.softplus(self.enc_std(enc_t, edge_index, edge_weight))[[target_i,target_j]]

            #encoder of supplement
            enc_d_t = F.relu(self.sup_enc(phi_d_t, edge_index_sup, edge_weight_sup))
            enc_d_mean_t = self.sup_enc_mean(enc_d_t, edge_index_sup, edge_weight_sup)
            
            #decoder
            z_t_in = self.phi_z_in(torch.cat([enc_mean_t, enc_d_mean_t], 1)[target_j])
            z_t_out = self.phi_z_out(torch.cat([enc_mean_t, enc_d_mean_t], 1)[target_i])
            z_t = torch.cat([z_t_out, z_t_in], dim=0).view((1, -1))
            dec_t = F.sigmoid(self.phi_dec(z_t)).view((1))

        return torch.exp(-iota * ((dec_t - truths[target_t][target_i][target_j]) ** 2))\
                - 0.01 * torch.norm(enc_std_t)

    
    def masked_pisloss(self, inputs, truth, mask, A_scaler):
        inputs = A_scaler.inverse_transform(inputs)
        #print(inputs.min(), inputs.max())
        theloss = self.pisloss(inputs, truth)
        #print(((theloss * mask) / (mask.sum())).sum())
        #print(((self.pisloss(truth, truth) * mask) / (mask.sum())).sum())
        return 11 * ((theloss * mask) / (mask.sum())).sum()
    
    def adj_to_index(self, adj):
        edge_index = []
        edge_weight = []
        for i in range(adj.size()[0]):
            for j in range(adj.size()[1]):
                if adj[i][j]>0:
                    edge_index.append([i,j])
                    edge_weight.append(adj[i][j])
        return torch.from_numpy(np.array(edge_index).T), torch.from_numpy(edge_weight)

    def index_to_adj(self, edge_index, edge_weight):
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype="float32")
        for i in range(edge_weight.size(0)):
            adj[edge_index[0][i]][edge_index[1][i]] += edge_weight[i]
        return torch.from_numpy(adj)


    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = torch.tensor(eps1, device = self.device, requires_grad=True)
        #eps1 = eps1.clone().detach().requires_grad_(True)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)