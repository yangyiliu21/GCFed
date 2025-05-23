import numpy as np
import cvxpy as cp
import torch
from optimization import optimization, optimization_1d
import copy
from torch import nn
from test_call_matlab import matlab_solve_noise, matlab_solve_noise_3user, matlab_solve_rate


def select_layers(args):

    if (args.model == 'mlp') and (args.dataset == 'mnist'):
        layers = ['layer_input.weight', 'layer_hidden.weight']
    elif (args.model == 'cnn') and (args.dataset == 'mnist'):
        layers = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
    elif (args.model == 'cnn') and (args.dataset == 'cifar'):  
        layers = ['conv1.weight', 'conv2.weight', 'fc1.weight','fc2.weight','fc3.weight']  
    elif (args.model == 'denser_cnn') and (args.dataset == 'cifar'):  
        layers = ['fc1.weight','fc2.weight']
    elif (args.model == 'cnn') and (args.dataset == 'fashion_mnist'):  
        layers = ['fc1.weight','fc2.weight']


        # ?



    elif (args.model == 'vgg16') and (args.dataset == 'cifar'):  
        #layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight']          # 'classifier.0.weight'     'classifier.3.weight'   'classifier.6.weight'
        layers = ['classifier.0.weight']
        if args.user_select:
            layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight'] 
        if args.simulate_quant:
            layers = ['classifier.0.weight']

        if (args.user_select==None) and (args.quant_method==None):
            layers = ['classifier.0.weight', 'classifier.3.weight', 'classifier.6.weight'] 

    return layers


def cov_analysis(args, clients_index_array, m, w_locals, w_glob):

    layers = select_layers(args)

    sample_per_layer = 100

    #idxs_users = np.random.choice(clients_index_array, m, replace=False)[0:2]
    idxs_users = np.arange(0, args.num_users, 1, dtype=int)

    covariance = {}

    for layer in layers:
        samples = torch.zeros(len(idxs_users), sample_per_layer)
        # determine the sample index for each layer. Same index for each layer for each user
        num_total = len(w_glob[layer].view(-1))
        indices = torch.randperm(num_total)[:sample_per_layer]
        for user_id in idxs_users:
            samples[user_id, :] = (w_locals[user_id][layer] - w_glob[layer]).view(-1)[[indices.long()]]
        covariance[layer] = torch.cov(samples)

    return covariance



def user_selection(covariance):

    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    interval = 3
    ITER_MAX = 2

    index = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        index[key] = optimization(sig_X, mul, STEP, NUM_SILENT, interval, ITER_MAX)

    return index




def cov_select(sig_X, user_num_select):

    sig_X = torch.from_numpy(sig_X).float().cuda()

    user_num_total = sig_X.shape[0]
    #user_num_select = 3

    def combinations(array, tuple_length, prev_array=[]):
        if len(prev_array) == tuple_length:
            return [prev_array]
        combs = []
        for i, val in enumerate(array):
            prev_array_extended = prev_array.copy()
            prev_array_extended.append(val)
            combs += combinations(array[i+1:], tuple_length, prev_array_extended)
        return combs

    user_list = list(range(0,user_num_total))
    comb = combinations(user_list, user_num_select)

    comb_matrix = np.array(comb)

    add_matrix = np.zeros((comb_matrix.shape[0], user_num_total))

    for row in range(0, comb_matrix.shape[0]):
        vec = add_matrix[row]
        vec[comb_matrix[row]] = 1
        add_matrix[row] = vec
    add_matrix = np.vstack([add_matrix, np.ones((1,user_num_total))])

    subtract_matrix = np.eye(comb_matrix.shape[0])
    subtract_matrix = np.hstack([subtract_matrix, -1*np.ones((comb_matrix.shape[0],1))])

    add_matrix = torch.from_numpy(add_matrix).float().to(sig_X.device)
    subtract_matrix= torch.from_numpy(subtract_matrix).float().to(sig_X.device)

    distortion = (subtract_matrix @ add_matrix) @ sig_X @ (subtract_matrix @ add_matrix).T

    min_distortion = torch.min(torch.diag(distortion))
    min_index = torch.argmin(torch.diag(distortion))
    assert torch.diag(distortion)[min_index] == min_distortion

    user_select = comb_matrix[min_index]

    return user_select


def cov_selection(covariance, user_num_select):

    index = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        index[key] = cov_select(sig_X*mul, user_num_select)

    return index




def distortion_selection(w, size, args, indexes, NUM_SILENT):

    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in indexes.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in indexes.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        w_avg[k] = torch.div(w_avg[k], totalSize)
    
    subtotalSize = 0
    w_miss = {}
    l2_distortion = nn.MSELoss()
    for k in indexes.keys():
        distortion = {}
        ids = []
        for i in range(0, len(w)):
            w_miss[i] = torch.mul(w_avg[k], totalSize) - (w[i][k] * size[i])
            subtotalSize = totalSize - size[i]
            w_miss[i] = torch.div(w_miss[i], subtotalSize)
            distortion[i] = l2_distortion(w_miss[i], w_avg[k])
        for ii in range(0, NUM_SILENT):
            id = max(distortion, key=distortion.get)
            distortion.pop(id, None)
            ids.append(id)
        indexes[k] = ids


    return indexes




def noise_genneration(covariance):

    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    distortion_ratio = 0.1
    ITER_MAX = 2

    noise = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        noise_diags, rate, distortion = optimization_1d(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
        noise[key] = [noise_diags, [rate, distortion]]

    return noise


def noise_generation_matlab(covariance, args):

    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    distortion_ratio = 0.1
    ITER_MAX = 2

    noise = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        noise_bin, A_bin, upper_bound, d_bin, noise_nobin_min, A_nobin_min, d_nobin_min, noise_cental, A_central = matlab_solve_noise(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
        if args.quant_method == 'binning':
            noise[key] = [noise_bin, A_bin, upper_bound, d_bin]
        elif args.quant_method == 'no_bin':
            noise[key] = [noise_nobin_min, A_nobin_min, upper_bound, d_nobin_min]
        elif args.quant_method == 'centralized':
            noise[key] = [noise_cental, A_central, upper_bound, 0]

    return noise


def noise_generation_matlab_both(covariance, args):
    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    distortion_ratio = 0.1
    ITER_MAX = 2

    noise = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        noise_bin, A_bin, upper_bound, d_bin, noise_nobin_min, A_nobin_min, d_nobin_min, noise_cental, A_central = matlab_solve_noise(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
        noise['bin'] = [noise_bin, A_bin, upper_bound, d_bin]
        noise['no_bin'] = [noise_nobin_min, A_nobin_min, upper_bound, d_nobin_min]

    return noise


def rate_calculation_matlab(covariance, args):
    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    distortion_ratio = 0.1
    ITER_MAX = 2

    rate = {}
    for key, value in covariance.items(): 
        sig_X = value.detach().cpu().numpy()
        max_num = np.max(np.abs(sig_X)).item()
        mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
        #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

        R_bin, R_nobin = matlab_solve_rate(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
        rate[key] = [R_bin, R_nobin]

    return rate




def noise_generation_matlab_3user(covariance, args):

    #T=list(covariance.values())[0].shape[0]
    STEP=0.99
    NUM_SILENT = 3
    distortion_ratio = 0.1
    ITER_MAX = 2

    noise = {}
    # for key, value in covariance.items(): 
    sig_X = covariance.detach().cpu().numpy()
    max_num = np.max(np.abs(sig_X)).item()
    mul = float('1e'+str(int(str(np.format_float_scientific(max_num)).split("e-")[-1]) -1))
    #mul = np.power(10, int(str(max_num).split("e-")[-1]) -1)

    noise_bin, A_bin, upper_bound, d_bin, noise_nobin_min, A_nobin_min, d_nobin_min, noise_cental, A_central = matlab_solve_noise_3user(sig_X, mul, STEP, NUM_SILENT, distortion_ratio, ITER_MAX)
    if args.quant_method == 'binning':
        noise = [noise_bin, A_bin, upper_bound, d_bin]
    elif args.quant_method == 'no_bin':
        noise = [noise_nobin_min, A_nobin_min, upper_bound, d_nobin_min]
    elif args.quant_method == 'centralized':
        noise = [noise_cental, A_central, upper_bound, 0]

    return noise





def topk_analysis(cov_normalized, corr_matrix):

    softmax = torch.nn.Softmax(dim=0)

    choose = "variance_based"
    choose = "rho_based"
    
    if choose == "variance_based":
        # choose X1, X2, X3 with largest variances
        vars, indices = torch.topk(torch.diag(cov_normalized), 3)

        # use X2 and X3 to estimate X1
        idx2 = indices[0].item()
        idx3 = indices[1].item()
        idx1 = indices[2].item()
        rho12 = corr_matrix[idx1, idx2]
        rho13 = corr_matrix[idx1, idx3]
        rho23 = corr_matrix[idx2, idx3]

        L1 = -(rho12-rho13*rho23)/(torch.square(rho23)-1)
        L2 = (rho12*rho23-rho13)/(torch.square(rho23)-1)            # could be negative because rho13 very small, causing rho12*rho23-rho13>0

        softmax_res = softmax(torch.Tensor([L1,L2]))                #[0.5312, 0.4688]

    elif choose == "rho_based":
        # choose X3, X1 with largest rhos with X2, X2 with the largest variance
        max_user = torch.argmax(torch.diag(cov_normalized))
        rhos, indices = torch.topk(corr_matrix[max_user,:], 3)
        idx2 = indices[0].item()
        idx3 = indices[1].item()
        idx1 = indices[2].item()    
        rho12 = corr_matrix[idx1, idx2]
        rho13 = corr_matrix[idx1, idx3]
        rho23 = corr_matrix[idx2, idx3]

        L1 = -(rho12-rho13*rho23)/(torch.square(rho23)-1)
        L2 = (rho12*rho23-rho13)/(torch.square(rho23)-1) 

        z = torch.Tensor([L1, L2])
        softmax_res = torch.exp(z) / torch.sum(torch.exp(z))        #[0.5998, 0.4002]

    user_idx = [idx2, idx3, idx1]


    return user_idx, softmax_res




















# def cov_analysis(args, clients_index_array, m, w_locals, w_glob):

#     layers = select_layers(args)

#     sample_per_layer = 100

#     #idxs_users = np.random.choice(clients_index_array, m, replace=False)[0:2]
#     idxs_users = np.arange(0, args.num_users, 1, dtype=int)

#     # determine the sample index for each layer. Same index for each layer for each user
#     indices = torch.zeros(len(layers), sample_per_layer)
#     for layer_num in range(len(layers)):
#         layer = layers[layer_num]
#         num_total = len(w_locals[0][layer].view(-1))
#         indices[layer_num] = torch.randperm(num_total)[:sample_per_layer]

#     samples = torch.zeros(len(idxs_users), len(layers), sample_per_layer)

#     for user_id in range(len(idxs_users)): 
#         for layer_num in range(len(layers)):
#             layer = layers[layer_num]
#             #samples[user_id, layer_num, :] = w_locals[idxs_users[user_id]][layer].view(-1)[[indices[layer_num].long()]]
#             samples[user_id, layer_num, :] = (w_locals[idxs_users[user_id]][layer] - w_glob[layer]).view(-1)[[indices[layer_num].long()]]

#     samples = torch.swapaxes(samples, 0, 1)
#     samples = samples.reshape(-1, sample_per_layer)
#     covariance = torch.cov(samples)    


#     return covariance



        # # sample how many? Per-layer? How to calculate the correlation? 
        # if (args.model == 'mlp') and (args.dataset == 'mnist'): # 'layer_input.weight'  'layer_hidden.weight'

        #     layers = ['layer_input.weight', 'layer_hidden.weight']
        #     sample_per_layer = 100

        #     idxs_users = np.random.choice(clients_index_array, m, replace=False)[0:2]

        #     # determine the sample index for each layer. Same index for each layer for each user
        #     indices = torch.zeros(len(layers), sample_per_layer)
        #     for layer_num in range(len(layers)):
        #         layer = layers[layer_num]
        #         num_total = len(w_locals[0][layer].view(-1))
        #         indices[layer_num] = torch.randperm(num_total)[:sample_per_layer]

        #     samples = torch.zeros(len(idxs_users), len(layers), sample_per_layer)

        #     for user_id in range(len(idxs_users)): 
        #         for layer_num in range(len(layers)):
        #             layer = layers[layer_num]
        #             samples[user_id, layer_num, :] = w_locals[idxs_users[user_id]][layer].view(-1)[[indices[layer_num].long()]]

        #     samples = samples.reshape(-1, sample_per_layer)
        #     covariance = torch.cov(samples)
        #     pass


        #     # for layer in ['layer_input.weight', 'layer_hidden.weight']:
        #     #     num_total = len(w_locals[0][layer].view(-1))
        #     #     #num_items = int( np.floor(num_total/ 10000))
        #     #     num_items = 5
        #     #     indices = torch.randperm(num_total)[:num_items]
        #     #     samples = torch.zeros(len(indices), len(idxs_users))
        #     #     #samples0 = w_locals[0]['layer_input.weight'].view(-1)[indices]  
        #     #     #samples1 = w_locals[1]['layer_input.weight'].view(-1)[indices]
        #     #     for index in range(len(indices)):
        #     #         for user_id in range(len(idxs_users)):
        #     #             samples[index, user_id] = w_locals[idxs_users[user_id]][layer].view(-1)[indices[index]]
        #     #     covariance = torch.cov(samples)

        # elif (args.model == 'cnn') and (args.dataset == 'mnist'):
        #     # for layer in ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']:              # 250, 5000, 16000, 500
        #     #     num_total = len(w_locals[0][layer].view(-1))
        #     #     #num_items = int( np.floor(num_total/ 10000))
        #     #     num_items = 5
        #     #     indices = torch.randperm(num_total)[:num_items]
        #     #     samples = torch.zeros(len(indices), len(idxs_users))
        #     #     for index in range(len(indices)):
        #     #         for user_id in range(len(idxs_users)):
        #     #             samples[index, user_id] = w_locals[idxs_users[user_id]][layer].view(-1)[indices[index]]
        #     #     covariance = torch.cov(samples)


        #     #     # cov[layer] = covariance matrix between samples0 and samples1

        #     layers = ['conv1.weight', 'conv2.weight', 'fc1.weight', 'fc2.weight']
        #     #layers = ['conv1.weight', 'conv2.weight']
        #     #layers = ['fc1.weight', 'fc2.weight']
        #     sample_per_layer = 100

        #     idxs_users = np.random.choice(clients_index_array, m, replace=False)[0:2]

        #     # determine the sample index for each layer. Same index for each layer for each user
        #     indices = torch.zeros(len(layers), sample_per_layer)
        #     for layer_num in range(len(layers)):
        #         layer = layers[layer_num]
        #         num_total = len(w_locals[0][layer].view(-1))
        #         indices[layer_num] = torch.randperm(num_total)[:sample_per_layer]

        #     samples = torch.zeros(len(idxs_users), len(layers), sample_per_layer)

        #     for user_id in range(len(idxs_users)): 
        #         for layer_num in range(len(layers)):
        #             layer = layers[layer_num]
        #             samples[user_id, layer_num, :] = w_locals[idxs_users[user_id]][layer].view(-1)[[indices[layer_num].long()]]

        #     samples = samples.reshape(-1, sample_per_layer)
        #     covariance = torch.cov(samples)
        #     pass

        # elif (args.model == 'cnn') and (args.dataset == 'cifar'):    
        #     layers = ['conv1.weight', 'conv2.weight', 'fc1.weight','fc2.weight','fc3.weight']    
        #     #        64*3*5*5=4800, 64*64*5*5=102400, 384*1024=393216, 192*384=73728,  192*10=1920
        #     layers = ['layers.0.weight', 'layers.1.weight', 'layers.2.weight','layers.3.weight','fc.weight'] 
        #     #  32*3*3*3=864,  32*32*3*3=9216,  32*32*3*3=9216,  32*32*3*3=9216, 2880
        #     sample_per_layer = 100
        #     idxs_users = np.random.choice(clients_index_array, m, replace=False)[0:2]

        #     # determine the sample index for each layer. Same index for each layer for each user
        #     indices = torch.zeros(len(layers), sample_per_layer)
        #     for layer_num in range(len(layers)):
        #         layer = layers[layer_num]
        #         num_total = len(w_locals[0][layer].view(-1))
        #         indices[layer_num] = torch.randperm(num_total)[:sample_per_layer]

        #     samples = torch.zeros(len(idxs_users), len(layers), sample_per_layer)

        #     for user_id in range(len(idxs_users)): 
        #         for layer_num in range(len(layers)):
        #             layer = layers[layer_num]
        #             samples[user_id, layer_num, :] = w_locals[idxs_users[user_id]][layer].view(-1)[[indices[layer_num].long()]]

        #     samples = samples.reshape(-1, sample_per_layer)
        #     covariance = torch.cov(samples)