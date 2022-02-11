# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2021 xxx. All Rights Reserved
#
################################################################################
"""
Description: The main training process for VMR-GAE on NYC dataset
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


def prep_env(flag='train'):
    # type: (str) -> dict
    """
    Desc:
        Prepare the environment
    Args:
        flag: specify the environment, 'train' or 'evaluate'
    Returns:
        A dict indicating the environment variables
    """
    parser = \
        argparse.ArgumentParser(description='{} [VMR-GAE] on the task of OD Matrix Completion'
                                .format("Training" if flag == "train" else "Evaluating"))
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
    if flag == "train":
        parser.add_argument('--iter_num', type=int, default=10000, help='The number of iterations')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='delay to apply kld_loss')
        parser.add_argument('--result_path', type=str, default='./nyc/results', help='result path')
    else:
        parser.add_argument('--sample_time', type=int, default=10, help='The sample time for point estimation')

    args = parser.parse_args()

    if flag == "train":
        if not os.path.exists(args.checkpoints):
            os.makedirs(args.checkpoints)
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
    else:
        if not os.path.exists(args.checkpoints):
            print('Checkpoint does not exist.')
            exit()

    #read data from file
    all_len = 84
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
    device = torch.device('cuda:0')
    mask = np.zeros((args.num_nodes, args.num_nodes))
    for i in range(args.timelen):
        the_adj = utils.index_to_adj_np(A_flow[i][0],A_flow[i][1], args.num_nodes)
        mask[np.where(the_adj>(2/max_value))] = 1.0
    mask = torch.tensor(mask, dtype=torch.bool).to(device)
    x = torch.eye(args.num_nodes).to(device)
    truths = torch.from_numpy(truths).to(device)
    for i in range(args.timelen):
        green_flow[i][0] = torch.tensor(green_flow[i][0]).T.to(device)
        green_flow[i][1] = torch.tensor(green_flow[i][1]).to(device)
        A_flow[i][0] = torch.tensor(A_flow[i][0]).T.to(device)
        A_flow[i][1] = torch.tensor(A_flow[i][1]).to(device)

    res = {
        "args": args,
        "primary_scale": A_scaler, "x": x,
        "mask": mask,
        "ground_truths": truths,
        "supple_data": green_flow,
        "train_data": A_flow, "val_data": val_data, "test_data": test_data
    }
    return res


if __name__ == '__main__':
    device = torch.device('cuda:0')
    env = prep_env()

    # load VMR-GAE and run
    model = VMR_GAE(x_dim=env["x"].shape[-1], h_dim=env["args"].hidden_dim,
                num_nodes=env["args"].num_nodes, n_layers=env["args"].rnn_layer, device=device,
                eps=1e-10, align=env["args"].align, is_region_feature=env["args"].x_feature) 
    model = model.to(device)  

    # Before training, read the checkpoints if available
    if not os.path.isfile('%s/model.pth' % env["args"].checkpoints):
        print("Start new train (model).")
        min_loss = np.Inf
        epoch = 0
    else:
        print("Found the model file. continue to train ... ")
        model.load_state_dict(torch.load('%s/model.pth' % env["args"].checkpoints).state_dict())
        min_loss = torch.load('%s/minloss.pt' % env["args"].checkpoints)
        epoch = np.load('%s/logged_epoch.npy' % env["args"].checkpoints)

    # initialize the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=env["args"].learning_rate)
    if os.path.isfile('%s/opt_state.pt' % env["args"].checkpoints):
        opt_state = torch.load('%s/opt_state.pt' % env["args"].checkpoints)
        optimizer.load_state_dict(opt_state)
    patience = 50
    best_val_mape = np.Inf
    max_iter = 0

    # start the training procedure
    for k in range(epoch, env["args"].iter_num):
        optimizer.zero_grad()
        kld_loss_tvge, kld_loss_avde, pis_loss, all_h, all_enc_mean, all_prior_mean, all_enc_d_mean, all_dec_t, \
            all_z_in, all_z_out \
            = model(env["x"], env["train_data"], env["supple_data"], env["mask"],
                    env["primary_scale"], env["ground_truths"])
        pred = env["primary_scale"].inverse_transform(all_dec_t[-1].cpu().detach().numpy())
        val_MAE, val_RMSE, val_MAPE = utils.validate(pred, env["val_data"][0],
                                                     env["val_data"][1], flag='val')
        test_MAE, test_RMSE, test_MAPE = utils.validate(pred, env["test_data"][0],
                                                        env["test_data"][1], flag='test')
        # train_MAE, train_RMSE, train_MAPE = utils.validate(pred, env["train_data"][0],
        #                                                    env["train_data"][1], flag='train')
        if val_MAPE < best_val_mape:
            best_val_mape = val_MAPE
            max_iter = 0
        else:
            max_iter += 1
            if max_iter >= patience:
                print('Early Stop!')
                break
        if k >= env["args"].delay:
            loss = kld_loss_tvge + kld_loss_avde + pis_loss
        else:
            loss = pis_loss
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
            print('epoch: ', k)
            print('loss =', loss.mean().item())
            print('kld_loss_tvge =', kld_loss_tvge.mean().item())
            print('kld_loss_avde =', kld_loss_avde.mean().item())
            print('pis_loss =', pis_loss.mean().item())
            print('val', "MAE:", val_MAE, 'RMSE:', val_RMSE, 'MAPE:', val_MAPE)
            print('test', "MAE:", test_MAE, 'RMSE:', test_RMSE, 'MAPE:', test_MAPE)

        if (loss.mean() < min_loss).item() | (k == env["args"].delay):
            print('epoch: %d, Loss goes down, save the model. pis_loss = %f' % (k, pis_loss.mean().item()))
            print('val', "MAE:", val_MAE, 'RMSE:', val_RMSE, 'MAPE:', val_MAPE)
            print('test', "MAE:", test_MAE, 'RMSE:', test_RMSE, 'MAPE:', test_MAPE)
            min_loss = loss.mean().item()
            torch.save(all_enc_mean, '%s/all_enc_mean.pt' % env["args"].result_path)
            torch.save(all_prior_mean, '%s/all_prior_mean.pt' % env["args"].result_path)
            torch.save(all_enc_d_mean, '%s/all_enc_d_mean.pt' % env["args"].result_path)
            torch.save(all_dec_t, '%s/all_dec_t.pt' % env["args"].result_path)
            torch.save(all_z_in, '%s/all_z_in.pt' % env["args"].result_path)
            torch.save(all_z_out, '%s/all_z_out.pt' % env["args"].result_path)
            torch.save(model, '%s/model.pth' % env["args"].checkpoints)
            torch.save(loss.mean(), '%s/minloss.pt' % env["args"].checkpoints)
            torch.save(optimizer.state_dict(), '%s/opt_state.pt' % env["args"].checkpoints)
            np.save('%s/logged_epoch.npy' % env["args"].checkpoints, k)
