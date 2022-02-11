# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 xxx, Inc. All Rights Reserved
#
################################################################################
"""
Description: Evaluate the trained VMR-GAE model on NYC dataset with RMSE, MAE, and MAPE
Authors: xxx
Date:    2021/10/26
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from model.vmrgae import VMR_GAE
import utils as utils
from utils import MinMaxScaler
from train import prep_env


if __name__ == '__main__':
    device = torch.device('cuda:0')
    env = prep_env(flag='eval')

    # load VMR-GAE and run
    model = VMR_GAE(x_dim=env["x"].shape[-1], h_dim=env["args"].hidden_dim,
                num_nodes=env["args"].num_nodes, n_layers=env["args"].rnn_layer, device=device,
                eps=1e-10, align=env["args"].align, is_region_feature=env["args"].x_feature) 
    model = model.to(device)

    if not os.path.isfile('%s/model.pth' % env["args"].checkpoints):
        print('Checkpoint does not exist.')
        exit()
    else:
        model.load_state_dict(torch.load('%s/model.pth' % env["args"].checkpoints).state_dict())
        min_loss = torch.load('%s/minloss.pt' % env["args"].checkpoints)
        epoch = np.load('%s/logged_epoch.npy' % env["args"].checkpoints)

    pred = []
    for i in range(env["args"].sample_time):
        _, _, _, _, _, _, _, all_dec_t, _, _ \
            = model(env["x"], env["train_data"], env["supple_data"], env["mask"],
                    env["primary_scale"], env["ground_truths"])
        pred.append(env["primary_scale"].inverse_transform(all_dec_t[-1].cpu().detach().numpy()))
    pred = np.stack(pred, axis=0)
    pe, std = pred.mean(axis=0), pred.std(axis=0)
    pe[np.where(pe < 0.5)] = 0
    print(pe)
