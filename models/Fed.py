#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn

#def FedWeightAvg(w, size, args, indexes=[]):
def FedWeightAvg(w, size, args):

    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)

    distortion = {}

    # # partial fedavg
    # if args.user_select:
    #     w_avg_ori = {}
    #     l2_distortion = nn.MSELoss()
    #     for k in indexes.keys():
    #         subtotalSize = 0
    #         w_avg_ori[k] = copy.deepcopy(w_avg[k])
    #         w_avg[k] = w_avg[k]*0
    #         for user_id in indexes[k]:
    #             w_avg[k] += w[user_id][k] * size[user_id]
    #             subtotalSize += size[user_id]
    #         w_avg[k] = torch.div(w_avg[k], subtotalSize)

    #         # calculate distortion
    #         distortion[k] = float(l2_distortion(w_avg[k], w_avg_ori[k]).cpu())

    # if args.user_select == False:
    #     totalSize = sum(size)                       # size is a list of how many local data each user has
    #     w_avg = copy.deepcopy(w[0])
    #     for k in w_avg.keys():
    #         w_avg[k] = w[0][k]*size[0]
    #     for k in w_avg.keys():
    #         for i in range(1, len(w)):
    #             w_avg[k] += w[i][k] * size[i]
    #         # print(w_avg[k])
    #         w_avg[k] = torch.div(w_avg[k], totalSize)

    #     distortion = {}

    # else:
    #     totalSize = sum(size)                       # size is a list of how many local data each user has
    #     w_avg = copy.deepcopy(w[0])
    #     w_avg_ori = {}
    #     for k in w_avg.keys():
    #         w_avg[k] = w[0][k]*size[0]
    #     for k in w_avg.keys():
    #         for i in range(1, len(w)):
    #             w_avg_ori[k] += w[i][k] * size[i]
    #             if i in indexes:
    #                 w_avg[k] += w[i][k] * size[i] * (args.num_users/len(indexes))
    #         # print(w_avg[k])
    #         w_avg[k] = torch.div(w_avg[k], totalSize)


            
    return w_avg


def FedWeightAvg_select(w, size, args, indexes):

    assert args.user_select, f"user_select not enabled"

    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)

    distortion = {}

    # partial fedavg
    w_avg_ori = {}
    num_param = {}
    total_param = 0
    l2_distortion = nn.MSELoss()
    for k in indexes.keys():

        num_param[k] = w[0][k].shape[0]*w[0][k].shape[1]
        total_param += num_param[k]

        subtotalSize = 0
        w_avg_ori[k] = copy.deepcopy(w_avg[k])
        w_avg[k] = w_avg[k]*0
        for user_id in indexes[k]:
            w_avg[k] += w[user_id][k] * size[user_id]
            subtotalSize += size[user_id]
        w_avg[k] = torch.div(w_avg[k], subtotalSize)

        # calculate distortion
        distortion[k] = float(l2_distortion(w_avg[k], w_avg_ori[k]).cpu())

    vanilla_sum = 0
    weighted_sum = 0
    for k in indexes.keys():
        vanilla_sum += distortion[k] * 1/len(indexes.keys())
        weighted_sum += distortion[k] * num_param[k]/total_param            # 0.859 0.140 0.000342
    
    distortion['sum'] = vanilla_sum
    distortion['weighetd_sum'] = weighted_sum
            
    return w_avg, distortion


def FedWeightAvg_noise(w, size, args, indexes, noise):

    assert args.simulate_quant==True, f"simulate_quant not enabled"

    # noise contains [noise_diags, [rate, distortion]]

    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)

    distortion = {}

    # partial fedavg
    #if args.user_select:
    if args.quant_method in ['binning', 'no_bin']:
        w_avg_true = {}
        w_avg_noised = {}
        w_avg_denoised = {}
        w_locals_noised = []
        l2_distortion = nn.MSELoss()
        for k in indexes.keys():
            rate_from_cov = noise[k][2]
            distortion_from_cov = noise[k][3]
            A_matrix = noise[k][1]
            subtotalSize = 0
            w_avg_true[k] = copy.deepcopy(w_avg[k])
            w_avg_noised[k] = copy.deepcopy(w_avg[k])
            w_avg_denoised[k] = copy.deepcopy(w_avg[k])
            w_avg[k] = w_avg[k]*0
            w_avg_noised[k] = w_avg_noised[k]*0
            w_avg_denoised[k] = w_avg_denoised[k]*0
            for user_id in indexes[k]:
                noise_var = torch.from_numpy(noise[k][0])[user_id]
                weight_shape = w[user_id][k].shape
                noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
                w_avg_noised[k] += (w[user_id][k]+noise_tensor.to(w[user_id][k].device)) * size[user_id]
                subtotalSize += size[user_id]
                w_locals_noised.append(w[user_id][k]+noise_tensor.to(w[user_id][k].device))
            # noised w_avg
            w_avg_noised[k] = torch.div(w_avg_noised[k], subtotalSize)

            # denoised w_avg
            subtotalSize = 0
            w_locals_denoised = []
            for user_id in indexes[k]:
                A_vec = A_matrix[user_id]
                denoise_sum = torch.zeros_like(w_locals_noised[0])
                for ii in indexes[k]:
                    denoise_sum += A_vec[ii] * w_locals_noised[ii]
                w_locals_denoised.append(denoise_sum)

                w_avg_denoised[k] += (denoise_sum) * size[user_id]
                subtotalSize += size[user_id]
            w_avg_denoised[k] = torch.div(w_avg_denoised[k], subtotalSize)

            # copy to w_avg[k]
            w_avg[k] = copy.deepcopy(w_avg_denoised[k])
            #w_avg[k] = copy.deepcopy(w_avg_noised[k])

            # calculate distortion
            distortion[k] = float(l2_distortion(w_avg[k], w_avg_true[k]).cpu())


    elif args.quant_method == 'centralized':
        #pass
        w_avg_true = {}
        w_avg_noised = {}
        w_avg_denoised = {}
        w_locals_noised = []
        l2_distortion = nn.MSELoss()
        for k in indexes.keys():
            rate_from_cov = noise[k][2]
            distortion_from_cov = noise[k][3]
            A_matrix = noise[k][1]
            subtotalSize = 0
            w_avg_true[k] = copy.deepcopy(w_avg[k])
            w_avg_noised[k] = copy.deepcopy(w_avg[k])
            w_avg_denoised[k] = copy.deepcopy(w_avg[k])
            w_avg[k] = w_avg[k]*0
            w_avg_noised[k] = w_avg_noised[k]*0
            w_avg_denoised[k] = w_avg_denoised[k]*0

            # add noise
            noise_var = torch.from_numpy(noise[k][0])           # should be single value
            #noise_var = noise_var*0
            weight_shape = w_avg[k].shape
            noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
            w_avg_noised[k] = w_avg_true[k] + noise_tensor.to(w_avg_true[k].device)

            # denoise
            #w_avg_denoised[k] = w_avg_noised[k]
            w_avg_denoised[k] = A_matrix * w_avg_noised[k]      # A should be single value, close to 1

            # copy to w_avg[k]
            w_avg[k] = copy.deepcopy(w_avg_denoised[k])

            # calculate distortion
            distortion['central'] = float(l2_distortion(w_avg[k], w_avg_true[k]).cpu())


    else:
        print('wrong quantization method simulated')
        exit(0)

    # if args.user_select == False:
    #     totalSize = sum(size)                       # size is a list of how many local data each user has
    #     w_avg = copy.deepcopy(w[0])
    #     for k in w_avg.keys():
    #         w_avg[k] = w[0][k]*size[0]
    #     for k in w_avg.keys():
    #         for i in range(1, len(w)):
    #             w_avg[k] += w[i][k] * size[i]
    #         # print(w_avg[k])
    #         w_avg[k] = torch.div(w_avg[k], totalSize)

    #     distortion = {}

    # else:
    #     totalSize = sum(size)                       # size is a list of how many local data each user has
    #     w_avg = copy.deepcopy(w[0])
    #     w_avg_ori = {}
    #     for k in w_avg.keys():
    #         w_avg[k] = w[0][k]*size[0]
    #     for k in w_avg.keys():
    #         for i in range(1, len(w)):
    #             w_avg_ori[k] += w[i][k] * size[i]
    #             if i in indexes:
    #                 w_avg[k] += w[i][k] * size[i] * (args.num_users/len(indexes))
    #         # print(w_avg[k])
    #         w_avg[k] = torch.div(w_avg[k], totalSize)


            
    return w_avg, distortion[k]





def FedWeightAvg_distortion(w, size, args, indexes, noise):

    assert args.simulate_quant==True, f"simulate_quant not enabled"

    # noise contains [noise_diags, [rate, distortion]]

    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)

    distortion = {}
    l2_distortion = nn.MSELoss()
    layer = 'classifier.0.weight'

    if args.quant_method in ['binning', 'no_bin']:
        #############################################  binning
        w_bin_locals_noised = []

        rate_bin= noise['bin'][2]
        distortion_bin = noise['bin'][3]
        A_matrix_bin = noise['bin'][1]

        w_bin_noised = copy.deepcopy(w_avg[layer])
        w_bin_denoised = copy.deepcopy(w_avg[layer])
        w_bin_noised = w_bin_noised*0
        w_bin_denoised = w_bin_denoised*0
        for user_id in indexes[layer]:
            noise_var = torch.from_numpy(noise['bin'][0])[user_id].float()
            weight_shape = w[user_id][layer].shape
            noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
            # w_bin_noised += (w[user_id][layer]+noise_tensor.to(w[user_id][layer].device)) * size[user_id]
            # subtotalSize += size[user_id]
            w_bin_locals_noised.append(w[user_id][layer]+noise_tensor.to(w[user_id][layer].device))
            # now we have 10 noised w_locals

        # denoised
        subtotalSize = 0
        #w_bin_locals_denoised = []
        for user_id in indexes[layer]:
            A_vec = A_matrix_bin[user_id]
            denoise_sum = torch.zeros_like(w_bin_locals_noised[0])
            for ii in indexes[layer]:
                denoise_sum += A_vec[ii] * w_bin_locals_noised[ii]              # number*vec?????
            #w_bin_locals_denoised.append(denoise_sum)

            w_bin_denoised += (denoise_sum) * size[user_id]
            subtotalSize += size[user_id]
        w_bin_denoised  = torch.div(w_bin_denoised, subtotalSize)

        # calculate distortion
        distortion['bin'] = float(l2_distortion(w_avg[layer], w_bin_denoised).cpu())




        #############################################  no_bin
        w_nobin_locals_noised = []

        rate_nobin= noise['no_bin'][2]
        distortion_nobin = noise['no_bin'][3]
        A_matrix_nobin = noise['no_bin'][1]

        w_nobin_noised = copy.deepcopy(w_avg[layer])
        w_nobin_denoised = copy.deepcopy(w_avg[layer])
        w_nobin_noised = w_nobin_noised*0
        w_nobin_denoised = w_nobin_denoised*0
        for user_id in indexes[layer]:
            noise_var = torch.from_numpy(noise['no_bin'][0])[user_id].float()
            weight_shape = w[user_id][layer].shape
            noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
            # w_bin_noised += (w[user_id][layer]+noise_tensor.to(w[user_id][layer].device)) * size[user_id]
            # subtotalSize += size[user_id]
            w_nobin_locals_noised.append(w[user_id][layer]+noise_tensor.to(w[user_id][layer].device))
            # now we have 10 noised w_locals

        # denoised
        subtotalSize = 0
        for user_id in indexes[layer]:
            A_vec = A_matrix_nobin[user_id]
            denoise_sum = torch.zeros_like(w_nobin_locals_noised[0])
            for ii in indexes[layer]:
                denoise_sum += A_vec[ii] * w_nobin_locals_noised[ii]           

            w_nobin_denoised += (denoise_sum) * size[user_id]
            subtotalSize += size[user_id]
        w_nobin_denoised  = torch.div(w_nobin_denoised, subtotalSize)

        # calculate distortion
        distortion['no_bin'] = float(l2_distortion(w_avg[layer], w_nobin_denoised).cpu())
    
    elif args.quant_method == 'centralized':
        # print("pass")
        w_cent_locals_noised = []

        rate_cent= noise[layer][2]
        distortion_cent = noise[layer][3]
        A_matrix_cent = noise[layer][1]

        w_cent_noised = copy.deepcopy(w_avg[layer])
        w_cent_denoised = copy.deepcopy(w_avg[layer])
        w_cent_noised = w_cent_noised*0
        w_cent_denoised = w_cent_denoised*0

        # add noise
        #noise_var = torch.from_numpy(noise[k][0])           # should be single value
        noise_var = torch.from_numpy(noise[layer][0]).float()      # should be single value
        #noise_var = noise_var*0
        weight_shape = w_avg[layer].shape
        noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
        w_cent_noised = copy.deepcopy(w_avg[layer]) + noise_tensor.to(w_avg[layer].device)

        # denoise
        #w_avg_denoised[k] = w_avg_noised[k]
        w_cent_denoised = A_matrix_cent * w_cent_noised      # A should be single value, close to 1

        # # copy to w_avg[k]
        # w_avg[k] = copy.deepcopy(w_avg_denoised[k])

        # calculate distortion
        distortion['central'] = float(l2_distortion(w_avg[layer], w_cent_denoised).cpu())


    elif args.quant_method == 'topk':

        max_user_idx = noise['no_bin']
        rhos = noise['bin']

        distortion = {}

        w_nobin = copy.deepcopy(w_avg[layer])*0
        w_nobin = w[max_user_idx][layer]

        distortion['no_bin'] = float(l2_distortion(w_avg[layer], w_nobin).cpu())

        if len(rhos) == 0:
            distortion
            return w_avg, distortion
        

        #  fedavg result = log( exp(1) + exp(0.8) )*user1_gradient 
        w_bin = copy.deepcopy(w_avg[layer])*0
        #w_bin = w[max_user_idx][layer]*(1 + torch.sum(rhos)) / (len(rhos)+1)
        w_bin = w[max_user_idx][layer]*(torch.log(torch.exp(torch.tensor([1]))+ torch.sum(torch.exp(rhos))).to(w[max_user_idx][layer].device))

        distortion['bin'] = float(l2_distortion(w_avg[layer], w_bin).cpu())

        # if using w_nobin or w_bin to do FedAVG
        # w_avg[layer] = copy.deepcopy(w_bin)
        # w_avg[layer] = copy.deepcopy(w_nobin)


    return w_avg, distortion




def FedWeightAvg_both(w, size, args, indexes, noise):


    totalSize = sum(size)                       # size is a list of how many local data each user has
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w[0][k]*size[0]
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k] * size[i]
        # print(w_avg[k])
        w_avg[k] = torch.div(w_avg[k], totalSize)

    distortion = {}
    device = w_avg['classifier.0.weight'].device
    l2_distortion = nn.MSELoss()

    # partial fedavg
    #if args.user_select:
    if args.quant_method in ['binning', 'no_bin']:
        for k in indexes.keys():
            w_avg_true = {}
            w_avg_noised = {}
            w_avg_denoised = {}
            w_locals_noised = []
            w_locals_nonoise = []
            rate_from_cov = noise[k][2]
            distortion_from_cov = noise[k][3]
            A_matrix = noise[k][1]
            #subtotalSize = 0
            w_avg_true[k] = copy.deepcopy(w_avg[k])
            w_avg_noised[k] = copy.deepcopy(w_avg[k])
            w_avg_denoised[k] = copy.deepcopy(w_avg[k])
            w_avg[k] = w_avg[k]*0
            w_avg_noised[k] = w_avg_noised[k]*0
            w_avg_denoised[k] = w_avg_denoised[k]*0
            ii=0
            for user_id in indexes[k]:
                noise_var = torch.from_numpy(noise[k][0])[ii]
                weight_shape = w[user_id][k].shape
                noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
                #w_avg_noised[k] += (w[user_id][k]+noise_tensor.to(device)) * size[user_id]
                #subtotalSize += size[user_id]
                w_locals_noised.append(w[user_id][k]+noise_tensor.to(device))
                w_locals_nonoise.append(w[user_id][k])
                ii += 1
            # noised w_avg
            #w_avg_noised[k] = torch.div(w_avg_noised[k], subtotalSize)

            # denoised w_avg
            #subtotalSize = 0
            #w_locals_denoised = []
            #iii=0
            #for user_id in indexes[k]:
            for cc in range(len(indexes[k])):
                A_vec = A_matrix[cc]
                denoise_sum = torch.zeros_like(w_locals_noised[0])
                for mm in range(len(A_vec)):
                    denoise_sum += A_vec[mm] * w_locals_noised[mm]
                #w_locals_denoised.append(denoise_sum)

                w_avg_denoised[k] += (denoise_sum) #* size[user_id]
                #subtotalSize += size[user_id]
                #iii += 1
            w_avg_denoised[k] = torch.div(w_avg_denoised[k], len(w_locals_noised))

            # copy to w_avg[k]
            w_avg[k] = copy.deepcopy(w_avg_denoised[k])                                     # debug w_avg[k]
            #w_avg[k] = copy.deepcopy(w_avg_noised[k])


            #calculate no noise case
            w_nonoise = torch.div(torch.stack(w_locals_nonoise, dim=0).sum(dim=0), len(w_locals_noised))

            # calculate distortion
            distortion_noise = float(l2_distortion(w_avg[k], w_avg_true[k]).cpu())
            distortion_nonoise = float(l2_distortion(w_nonoise, w_avg_true[k]).cpu())
            distortion[k] = [distortion_noise, distortion_nonoise]

    elif args.quant_method == 'centralized':

        for k in indexes.keys():
            w_cent_locals_noised = []
            rate_cent= noise[k][2]
            distortion_cent = noise[k][3]
            A_matrix_cent = noise[k][1]

            #w_cent_noised = copy.deepcopy(w_avg[k])*0
            #w_cent_denoised = copy.deepcopy(w_avg[k])*0

            # add noise
            noise_var = torch.from_numpy(noise[k][0]).float()
            weight_shape = w_avg[k].shape
            noise_tensor = torch.randn(weight_shape[0], weight_shape[1]) * torch.sqrt(noise_var)
            w_cent_noised = copy.deepcopy(w_avg[k]) + noise_tensor.to(device)

            # denoise
            w_cent_denoised = A_matrix_cent * w_cent_noised 

            # calculate distortion
            distortion[k] = float(l2_distortion(w_avg[k], w_cent_denoised).cpu())

            # copy to w_avg[k]
            w_avg[k] = copy.deepcopy(w_cent_denoised)



    else:
        print('wrong quantization method simulated')
        exit(0)


            
    return w_avg, distortion


