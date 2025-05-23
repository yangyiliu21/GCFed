#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse




def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")                                         # learning rate 0.01
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")                      # 0.9 default=0.5
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    parser.add_argument('--bias', type=float, default=0.5, help="non iid setting")
    parser.add_argument('--shards_per_client', type = int, default=10, help='number of shards for each client in non iid setting')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # quantization arguments
    parser.add_argument('--R', type=int, default=0, help="rate")    #1
    parser.add_argument('--M', type=int, default=0, help="m value")
    parser.add_argument('--compression_type', type=str, default='no_compression', help="compression type")  #no_compression  GenNorm
    parser.add_argument('--sp_perc', type=int, default=100, help="sparsification percentage")


    # covariance analysis arguments
    parser.add_argument('--cov_analysis', action='store_true', help='whether do covariance analysis or not')

    # user selection arguments
    parser.add_argument('--user_select', choices=['random', 'cov', 'top', 'true_distortion'], help='how to perform user selection')
    parser.add_argument('--num_select', type=int, default=3, help="number of selected users: m")

    # quantization arguments
    parser.add_argument('--simulate_quant', action='store_true', help='whether simulate quantization or not')
    parser.add_argument('--quant_method', choices=['binning', 'no_bin', 'centralized', 'topk'], help='quantization method')


    # Compare baseline methods
    parser.add_argument('--gpr',action = 'store_true', help = 'Use FedCor')
    parser.add_argument('--GPR_interval',type = int, default= 1, help = 'interval of sampling and training of GP, namely, Delta t')
    
    parser.add_argument('--afl',action = 'store_true', help = 'use AFL selection')

    parser.add_argument('--power_d',action = 'store_true', help = 'use Pow-d selection')

    # Quantization + user_selection
    parser.add_argument('--both',action = 'store_true', help = 'Use both Quantization and user_selection')


    args = parser.parse_args()
    return args
