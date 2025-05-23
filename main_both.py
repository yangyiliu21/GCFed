#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms, models
import torch
import os

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid, cifar_noniid_shard
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM, SimpleConvNet, CNNCifar_denser, VGG16
from models.simple_models import Net, Net1, Net2, ResNet18, ResNet9, Net3
from models.Fed import FedWeightAvg, FedWeightAvg_noise, FedWeightAvg_select, FedWeightAvg_distortion, FedWeightAvg_both
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from utils.utility import top_k_sparsificate_model_weights, quantization

from cov_analysis import cov_analysis, cov_selection, distortion_selection, noise_genneration, noise_generation_matlab, noise_generation_matlab_3user, noise_generation_matlab_both

import baselines.GPR as GPR
from baselines.GPR import Kernel_GPR
from baselines.update import train_federated_learning, federated_test_idx

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

if __name__ == '__main__':

    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.manual_seed(SEED)
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    if args.gpr:
        baseline = 'gpr'
    elif args.afl:
        baseline = 'afl'
    elif args.power_d:
        baseline = 'pow_d'
    else:
        baseline = args.quant_method

    if args.both:
        baseline = 'both'
        args.cov_analysis = True
        args.user_select == 'cov'
        args.quant_method == True
        args.simulate_quant == False

    # # # manually set input arguments
    # args.dataset = 'fashion-mnist'
    # args.num_channels = 1
    # args.num_users = 3
    # args.frac = 1
    # args.model = 'cnn'
    # args.epochs = 10

    #print(str(args.bias)+'/'+args.compression_type+'_Rate'+str(args.R)+'_Mvalue'+str(args.M)+"_%"+str(args.sp_perc))

    #compression_type = 'GenNorm'
    #QUANTIZATION_M = 0

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        # TODO
        # create smaller dataset for 2-user or max 5-user

        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.bias)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            print("################################ ", baseline)
            #print("################################ ", args.quant_method)
            if args.shards_per_client == 10:
                assert args.lr==0.03, f"for bias noniid, change learning rate to 0.03"
                dict_users = cifar_noniid(dataset_train, args.num_users, args.bias)
            else:
                args.bias = args.shards_per_client
                assert args.lr==0.01, f"for shard noniid, change learning rate to 0.01"
                dict_users = cifar_noniid_shard(dataset_train, args.num_users, args.shards_per_client, np.random.RandomState(SEED))
    elif args.dataset == 'fashion-mnist':
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users, args.bias)
    elif args.dataset == 'femnist':
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = SimpleConvNet(args=args).to(args.device)      # SimpleConvNet  CNNCifar
    elif args.model == 'denser_cnn' and args.dataset == 'cifar':
        net_glob = Net1(args=args).to(args.device)       # SimpleConvNet  CNNCifar
    elif args.model == 'resnet' and args.dataset == 'cifar':
        net_glob = ResNet18().to(args.device)       # SimpleConvNet  CNNCifar
    elif args.model == 'vgg16' and args.dataset == 'cifar':
        #net_glob = VGG16().to(args.device)       # SimpleConvNet  CNNCifar
        net_glob = models.vgg16(weights='IMAGENET1K_V1')                        # IMAGENET1K_FEATURES, IMAGENET1K_V1   &   freeze other layers

        # initiate the classifier part from scratch
        # no_pretrained = models.vgg16()
        # net_glob.classifier = no_pretrained.classifier

        # modify the last layer for Cifar10 output categories
        input_lastLayer = net_glob.classifier[6].in_features
        torch.manual_seed(SEED)
        net_glob.classifier[6] = torch.nn.Linear(input_lastLayer,10)

        # input_secondlastLayer = net_glob.classifier[3].in_features
        # output_secondlastLayer = net_glob.classifier[3].out_features
        # net_glob.classifier[3] = torch.nn.Linear(input_secondlastLayer,output_secondlastLayer)

        net_glob = net_glob.to(args.device)
        for param in net_glob.parameters():
            param.requires_grad = False
        
        for param in net_glob.classifier.parameters():
            param.requires_grad = True
        # for i in range(6, 7):
        #     for param in net_glob.classifier[i].parameters():
        #         param.requires_grad = True
        for param in net_glob.avgpool.parameters():
            param.requires_grad = True
        for i in range(24, 31):
            for param in net_glob.features[i].parameters():
                param.requires_grad = True
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion_mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    # elif args.model == 'denser_cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
    #     net_glob = Net(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    
    #print(net_glob)                                 # TODO: correlation MLP vs. CNN
    print(str(args.bias)+'/'+args.compression_type+'_Rate'+str(args.R)+'_Mvalue'+str(args.M)+"_%"+str(args.sp_perc))
    net_glob.train()

    # Build GP
    if args.gpr:
        gpr = Kernel_GPR(args.num_users, loss_type='MML', reusable_history_length=1, gamma=0.99, device=args.device,
                        dimension = 15, kernel=GPR.Poly_Kernel, order = 1, Normalize = 0)
        gpr.to(args.device)

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    acc_test = []
    clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
               for idx in range(args.num_users)]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
    gt_global_losses = []
    for iter in range(args.epochs):
        w_locals, loss_locals, weight_locols= [], [], []

        #idxs_users = np.random.choice(clients_index_array, m, replace=False)
        idxs_users = np.sort(np.random.choice(clients_index_array, m, replace=False))

        for idx in idxs_users:
            w, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device))

            #TODO: for CNN, manipulate the weights for only 'conv2.weight' (5000) and 'fc1.weight'(16000)
            #TODO: sparsification + quantization
            # layer_name = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
            # for layer in layer_name:
            #     gradient = w[layer] - w_glob[layer]

            #     #print('sparse level:', sparsification_percentage/(100))
            #     #sparse_gradient = top_k_sparsificate_model_weights(gradient, args.sp_perc/(100))

            #     #quantized_gradient = quantization(sparse_gradient, args)

            #     #w[layer] = quantized_gradient + w_glob[layer]


            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))
        

        if args.gpr or args.afl or args.power_d:
            args.simulate_quant = False
            args.cov_analysis = False
            args.user_select = True
        # analyze correlation
        if args.cov_analysis:
            #clients sample a small number of gradients and send to server, server analyzes the correlation
            covariance = cov_analysis(args, clients_index_array, m, w_locals, net_glob.state_dict())   
            #print(covariance)

            if args.user_select:
                #num_select = 3
                #server assign bits to client, pick up the 3 users with the most bits
                if args.user_select == 'cov':
                    indexes = cov_selection(covariance, args.num_select)
                    #indexes_true = distortion_selection(w_locals, weight_locols, args, indexes, num_select)
                    # for key, cov_matrix in covariance.items(): 
                    #     indexes[key] = cov_selection(cov_matrix, num_select)

                elif args.user_select == 'random':
                    indexes = {}
                    #ids = np.random.choice(clients_index_array, m, replace=False)[:NUM_SILENT]
                    for key, _ in covariance.items(): 
                        indexes[key] = np.random.choice(clients_index_array, m, replace=False)[:args.num_select]         #[6,1,4] [1,5,6] [1,9,4]
                        #indexes[key] = ids
                elif args.user_select == 'true_distortion':
                    indexes = {}
                    for key, _ in covariance.items(): 
                        indexes[key] = None
                    indexes = distortion_selection(w_locals, weight_locols, args, indexes, args.num_select)
                elif args.user_select == 'top':
                    indexes = {}
                    for key, cov_matrix in covariance.items(): 
                        topk_diags, idxes = torch.topk(torch.diag(cov_matrix), args.num_select)
                        indexes[key] = idxes.numpy().astype(np.int32)
            else:
                # full participation
                indexes = {}
                for key, _ in covariance.items(): 
                    indexes[key] = list(clients_index_array)
        elif args.gpr:
            gpr_weights = np.ones(args.num_users)*(1/args.num_users)
            idxs_users = gpr.Select_Clients(args.num_select, 0.0, gpr_weights, False, 0.0)
            print("GPR Chosen Clients:",idxs_users)
            layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight']
            indexes = {} 
            for layer in layers:
                indexes[layer] = np.array(idxs_users)
        elif args.afl:
            afl_weights = np.ones(args.num_users)*(1/args.num_users)
            if iter == 0:
                # Test the global model before training
                net_glob.eval()
                list_acc, list_loss = federated_test_idx(args, net_glob,
                                                list(range(args.num_users)),
                                                dataset_train, dict_users)
                AFL_Valuation = np.array(list_loss)*np.sqrt(afl_weights*len(dataset_train))         # len(list_loss)=100 

            alpha1 = 0.75
            alpha2 = 0.01
            alpha3 = 0.1
            delete_num = int(alpha1*args.num_users)                    # 0.75*100
            sel_num = int((1-alpha3)*args.num_select)                                # (1-0.1)*5
            tmp_value = np.vstack([np.arange(args.num_users),AFL_Valuation])# [2, 100]
            tmp_value = tmp_value[:,tmp_value[1,:].argsort()]               
            prob = np.exp(alpha2*tmp_value[1,delete_num:])             # args.alpha2=0.01      (25,)
            prob = prob/np.sum(prob)
            sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
            remain = set(np.arange(args.num_users))-set(sel1)
            sel2 = np.random.choice(list(remain), args.num_select-sel_num, replace = False)
            idxs_users = np.append(sel1,sel2)
            print("AFL Chosen Clients:",idxs_users)
            layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight']
            indexes = {} 
            for layer in layers:
                indexes[layer] = np.array(idxs_users)
        elif args.power_d:
            # Power-of-D-choice
            if iter == 0:
                # Test the global model before training
                net_glob.eval()
                list_acc, list_loss = federated_test_idx(args, net_glob,
                                                list(range(args.num_users)),
                                                dataset_train, dict_users)
                gt_global_losses.append(list_loss)
            d = 5
            powd_weights = np.ones(args.num_users)*(1/args.num_users)
            A = np.random.choice(range(args.num_users), d, replace=False, p=powd_weights)
            idxs_users = A[np.argsort(np.array(gt_global_losses[-1])[A])[-args.num_select:]]
            print("powd Chosen Clients:",idxs_users)
            layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight']
            indexes = {} 
            for layer in layers:
                indexes[layer] = np.array(idxs_users)




        # simulating quantization
        if args.quant_method:
            if args.user_select:
                # # user select
                # cov_matrix = covariance['classifier.0.weight']
                # cov_matrix = cov_matrix[indexes['classifier.0.weight'],:][:,indexes['classifier.0.weight']]
                # covariance_sub = {}
                # covariance_sub['classifier.0.weight'] = cov_matrix
                # noise = noise_generation_matlab_3user(covariance_sub, args)

                covariance_sub = {}
                noise = {}
                for key, value in covariance.items():
                    cov_matrix = covariance[key]
                    cov_submatrix = cov_matrix[indexes[key],:][:,indexes[key]]
                    covariance_sub[key] = cov_submatrix
                    noise_sub = noise_generation_matlab_3user(cov_submatrix, args)
                    noise[key] = noise_sub

                
            else:
                # full participation
                if args.quant_method in ['binning', 'no_bin']:
                    # we do binning and no_bin at the same time
                    # noise = noise_generation_matlab(covariance, args)  
                    noise = noise_generation_matlab_both(covariance, args)
                    print(f'bin: {np.sum(noise["bin"][0])}; no_bin: {np.sum(noise["no_bin"][0])}')
                elif args.quant_method == 'centralized':
                    noise = noise_generation_matlab(covariance, args)  
                elif args.quant_method == 'topk':
                    # normalize the covariance matrix and select the one with largest variance
                    cov_matrix = covariance['classifier.0.weight']
                    cov_normalized = torch.div(cov_matrix, torch.norm(cov_matrix, p='fro'))                     # sum_diagonal != 1
                    max_user = torch.argmax(torch.diag(cov_normalized))

                    # from cov_matrix to correlation matrix (all ones in diagonal)
                    Dinv = torch.diag(1 / torch.sqrt(torch.diag(cov_matrix)))
                    corr_matrix = Dinv @ cov_matrix @ Dinv
                    rhos = corr_matrix[max_user, :]
                    rhos = rhos[rhos < 0.99999]
                    threhold = 0.6
                    selected_rhos = rhos[rhos > threhold]
                    print("selected rhos: ", selected_rhos)
                    # selected_rhos, selected_rhos_idxes = torch.topk(rhos, 3)
                    noise = {}
                    noise['no_bin'] = max_user
                    noise['bin'] = selected_rhos

        # # select corresponding w_locals
        # if args.user_select:
        #     w_locals_select = []
        #     for user_id in indexes['classifier.0.weight']:
        #         w_locals_select.append(w_locals[user_id])
        #     w_locals = w_locals_select


        #### w_locals wrong. Do partial add in FedWeightAvg


        # update global weights 
        #w_glob = FedWeightAvg(w_locals, weight_locols, args)
        #w_glob, distortion = FedWeightAvg(w_locals, weight_locols, args, indexes)
        if args.simulate_quant:
            assert args.user_select==None, f"can't do simulate_quant and user_select at the same time"
            #w_glob, distortion = FedWeightAvg_noise(w_locals, weight_locols, args, indexes, noise)
            w_glob, distortion = FedWeightAvg_distortion(w_locals, weight_locols, args, indexes, noise)
        elif args.user_select and not args.both:
            assert args.simulate_quant==False, f"can't do simulate_quant and user_select at the same time" 
            w_glob, distortion = FedWeightAvg_select(w_locals, weight_locols, args, indexes)
        elif args.both:
            w_glob, distortion = FedWeightAvg_both(w_locals, weight_locols, args, indexes, noise)
        else:
            w_glob = FedWeightAvg(w_locals, weight_locols, args)
        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        print("Round {:3d},Testing accuracy: {:.2f}".format(iter, acc_t))
        print(distortion)
        # for layer in distortion.keys():
        #     print(layer, distortion[layer])
        # print("weighted sum, ", distortion['weighetd_sum'] )

        acc_test.append(acc_t.item())
        #gt_global_losses.append(loss_t)

        if args.afl or args.gpr or args.power_d:
            net_glob.eval()
            list_acc, list_loss = federated_test_idx(args, net_glob,
                                                list(range(args.num_users)),
                                                dataset_train, dict_users)
            gt_global_losses.append(list_loss)


        if args.afl:
            AFL_Valuation = np.array(list_loss)*np.sqrt(afl_weights*len(dataset_train))         # len(list_loss)=100    


        if args.gpr and iter%args.GPR_interval==0:
            # train GPR
            gpr.Reset_Discount()
            #print("Training with Random Selection For GPR Training:")
            random_idxs_users = np.random.choice(range(args.num_users), args.num_select, replace=False)
            gpr_acc,gpr_loss = train_federated_learning(args, iter,
                                copy.deepcopy(net_glob), random_idxs_users, dataset_train, dict_users)
            gpr.Update_Training_Data([np.arange(args.num_users),],[np.array(gpr_loss)-np.array(gt_global_losses[-1]),],epoch=iter)
            #print("Training GPR")
            gpr.Train(lr = 1e-2, llr = 0.01, max_epoches=100, schedule_lr=False, update_mean=True, verbose=0)  #args.verbose=1

            # # test
            # idxs_users = gpr.Select_Clients(args.num_select, 0.0, gpr_weights, False, 0.0)

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_bias{}_{}_{}_iid{}_{}_R{}_M{}_sp%{}.dat'.
                   format(args.dataset, args.bias, args.model, args.epochs, args.iid, args.compression_type, args.R, args.M, args.sp_perc), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # # plot loss curve
    # plt.figure()
    # plt.plot(range(len(acc_test)), acc_test)
    # plt.ylabel('test accuracy')
    # plt.savefig(rootpath + '/fed_{}_{}_{}_local{}_Bias{}_iid{}_acc.png'.format(args.dataset, args.model, args.epochs, args.local_ep, args.bias, args.iid))



