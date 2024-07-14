import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# import yaml
from utils.random_tools import fix_random
from prepare_dataset import BDDataset
import torch.utils.data as Data
from utils.training_tools import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')

    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--result', type=str, default='jingleback_resnet', help='the location of result')
    parser.add_argument('--record_layer', type=str, default='layer3.1.conv2.weight', help='the layer name for record')
    parser.add_argument('--data_type', choices=['clean_test','poison_test','clean_val'], default='clean_val', help='the unlearning data type')
    
    parser.add_argument('--val_ratio', type=float, default=0.05) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--layer_type', type=str, default='conv', help='the type of layer for reinitialization')
    
    parser.add_argument('--lr_un', type=float, default=0.0000001)
    parser.add_argument('--unlearn_epochs', type=int, default=500, help="Number of epochs for training")
    parser.add_argument('--reinit_weight_ratio', type=float, default=0.7)
    parser.add_argument('--lr_ft', type=float, default=0.001)  # ft_reg 0.001 ft 0.005
    parser.add_argument('--ft_epochs', type=int, default=101, help="Number of epochs for training")
    parser.add_argument('--r', type=float, default=0.05, help='the r for regularization')
    parser.add_argument('--alpha', type=float, default=0.7, help='the alpha for regularization')
    
    args = parser.parse_args()
    return args

def get_layerName_from_type(model, layer_type):
    if layer_type == 'conv':
        instance_name = nn.Conv2d
    elif layer_type == 'bn':
        instance_name = nn.BatchNorm2d
    else:
        raise SystemError('NO valid layer_type match!') ######################
    layer_names = []
    for name, module in model.named_modules():
        if isinstance(module, instance_name) and 'shortcut' not in name:
            layer_names.append(name+'.weight')
    return layer_names
    
def train_finetuning(model, criterion, optimizer, data_loader):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels, *additional_info) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            nb_samples += images.size(0)

            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)

            pred = output.data.max(1)[1]
            # print(pred)
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        # print('Ft acc:', acc*100)
        # print('Ft loss:', loss)
        return loss, acc
    
def train_finetuning_reg(model, criterion, optimizer, data_loader, r, alpha): ########################
    model.train()
    total_correct = 0
    total_loss = 0.0
    nb_samples = 0
    for i, (images, labels, *additional_info) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        nb_samples += images.size(0)

        model_temp = copy.deepcopy(model)
        out1 = model_temp(images)
        loss1 = criterion(out1, labels)
        model_temp.zero_grad()
        loss1.backward()
        g1 = [param.grad.data.clone() for param in model_temp.parameters()]

        with torch.no_grad():
            for param, grad in zip(model_temp.parameters(), g1):
                param.data += r * (grad/grad.norm())
        out2 = model_temp(images)
        loss2 = criterion(out2, labels)
        model_temp.zero_grad()
        loss2.backward()
        g2 = [param.grad.data.clone() for param in model_temp.parameters()]
            
        optimizer.zero_grad()
        final_gradients = [(1 - alpha) * g1_item + alpha * g2_item for g1_item, g2_item in zip(g1, g2)]
        for param, grad in zip(model.parameters(), final_gradients):
            if grad is not None:
                param.grad = grad
        optimizer.step()

        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / nb_samples
    return loss, acc, final_gradients

def temp_test(model, criterion, data_loader): #########################
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (inputs, labels, *add_info) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            # print(pred)
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def mixed_name(layer_name, idx):
    return layer_name+'.'+str(idx)

def get_neuron_weight_norm(model, params_o, target_layer_list):
    target_neurons_list = []
    weight_norm_list = []
    weight_dict = {}
    for layer_i in range(len(params_o['params'])):
        name_i  = params_o['names'][layer_i]
        if not name_i in target_layer_list:
            continue
        params_i = params_o['params'][layer_i]   # 当前参数层的变化
        weight_i = params_i.view(params_i.shape[0], -1)# 展开为二维，并取绝对值
        neuron_i = weight_i.sum(dim=-1)                               # 计算每个神经元的权重变化总和
        for idx in range(neuron_i.size(0)):
            neuron_name =  mixed_name(name_i, idx)
            target_neurons_list.append(neuron_name)
            # changed_values_weightOrder[neuron_name] = torch.argsort(changed_weight_i[idx], descending=True).tolist()
            weight_dict[neuron_name] = torch.norm(weight_i[idx], p=2, dim=-1).item()
            weight_norm_list.append(weight_dict[neuron_name])
    return weight_norm_list, target_neurons_list

def pruning(net, top_num, neurons_list):
    state_dict = net.state_dict()
    for neuron_info in neurons_list[:top_num]:
        parts = neuron_info.split('.')
        nidx = int(parts[-1])  # 获取神经元索引
        layer_name = '.'.join(parts[:-1])  # 获取层名称
        state_dict[layer_name][int(nidx)] = 0.0
    net.load_state_dict(state_dict)
    return net

def prune_one_neuron(net, layer_name, idx):
    state_dict = net.state_dict()
    state_dict[layer_name][idx] = 0.0
    net.load_state_dict(state_dict)
    return net

def get_loss_change(model, criterion, data_loader, neurons_list, org_loss):
    loss_change_list = []
    for neuron_info in tqdm(neurons_list):
        model_copy = copy.deepcopy(model)
        parts = neuron_info.split('.')
        nidx = int(parts[-1])  # 获取神经元索引
        layer_name = '.'.join(parts[:-1])  # 获取层名称
        model_pruned = prune_one_neuron(model_copy, layer_name, nidx)
        loss, _ = temp_test(model_pruned, criterion, data_loader)
        loss_change = loss - org_loss
        loss_change_list.append(loss_change)
    return loss_change_list
   
def normalize_and_invert(scores):
    scores = np.array(scores)
    max_val = np.max(scores)
    min_val = np.min(scores)
    normalized_scores = (scores - min_val) / (max_val - min_val)
    inverted_scores = 1 - normalized_scores
    return inverted_scores.tolist()     

def mitigation(args):
    fix_random()
    defense_save_path = 'record/' + args.result + '/defense/ft_reg/'
    if not os.path.exists(defense_save_path):
        os.makedirs(defense_save_path) 
        
    # 1. load clean val set clean test set bd test set
    clean_path = 'record/' + args.result + '/' + args.dataset + '/clean/'
    bd_path = 'record/' + args.result + '/' + args.dataset + '/bd/'
    clean_train_mfcc = np.load(clean_path + 'clean_train_mfcc.npy')
    clean_train_label = np.load(clean_path + 'clean_train_label.npy')
    clean_test_mfcc = np.load(clean_path + 'clean_test_mfcc.npy')
    clean_test_label = np.load(clean_path + 'clean_test_label.npy')
    bd_test_mfcc = np.load(bd_path + 'bd_test_mfcc.npy')
    bd_test_label = np.load(bd_path + 'bd_test_label.npy')
    bd_test_index = np.load(bd_path + 'poison_index_test.npy')
    indices = list(range(0, len(clean_train_mfcc)))
    val_indices = random.sample(indices, int(len(clean_train_mfcc)*args.val_ratio))
    clean_val_mfcc = []
    clean_val_label = []
    for index in range(0, len(clean_train_mfcc)):
        if index in val_indices:
            clean_val_mfcc.append(clean_train_mfcc[index])
            clean_val_label.append(clean_train_label[index])
    # print(clean_val_label)
    clean_val_set = Data.TensorDataset(torch.tensor(clean_val_mfcc).float(), torch.tensor(clean_val_label))
    clean_test_set = Data.TensorDataset(torch.tensor(clean_test_mfcc).float(), torch.tensor(clean_test_label))
    bd_test_set = Data.TensorDataset(torch.tensor(bd_test_mfcc).float(), torch.tensor(bd_test_label))
    bd_test_set_complete = BDDataset(torch.tensor(bd_test_mfcc).float(), torch.tensor(bd_test_label), torch.tensor(bd_test_index))
    print('Length of val set:', len(clean_val_set))
    print('Length of test set', len(clean_test_set))
    clean_val_loader = Data.DataLoader(dataset=clean_val_set, batch_size=args.batch_size, shuffle=True)
    bd_test_loader = Data.DataLoader(dataset=bd_test_set, batch_size=args.batch_size, shuffle=True)
    bd_test_loader_complete = Data.DataLoader(dataset=bd_test_set_complete, batch_size=args.batch_size, shuffle=True)
    clean_test_loader = Data.DataLoader(dataset=clean_test_set, batch_size=args.batch_size, shuffle=True)
    
    # 2. load and select model layer data
    model_path = 'record/' + args.result + '/checkpoint.pt'
    bd_model = torch.load(model_path, map_location=device)
    model = copy.deepcopy(bd_model)
    parameters_o = list(bd_model.named_parameters())
    target_layers = get_layerName_from_type(bd_model, args.layer_type)
    params_o = {'names':[n for n, v in parameters_o if n in target_layers],
                    'params':[v for n, v in parameters_o if n in target_layers]}
    criterion = nn.CrossEntropyLoss()
    _, val_acc = temp_test(model, criterion, clean_val_loader) # 用后门模型测试
    _, test_acc = temp_test(model, criterion, clean_test_loader)
    _, test_asr = temp_test(model, criterion, bd_test_loader)
    print(f"Test loaded model: acc_{100*test_acc}, asr_{100*test_asr}, val_acc_{100*val_acc}")
    
    is_finetune_reg = True
    model_copy = copy.deepcopy(bd_model)
    test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
    print(f"Test bd model: acc_{test_clean_acc}, asr_{test_asr}")
    lr = 0.01
    lr_mult = 0.9
    parameters = []
    for idx, name in enumerate(target_layers):
        print(f'{idx}: lr = {lr:.6f}, {name}')
        parameters += [{'params': [p for n, p in model_copy.named_parameters() if n == name and p.requires_grad], 'lr': lr}]
        lr *= lr_mult
    update_neuron_params_optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.lr_ft, momentum=0.9)
    update_neuron_params_optimizer_layerwise = torch.optim.SGD(parameters)
    pbar = tqdm(range(300)) # args.ft_epochs+1
    for epoch in pbar: # finetune
        if is_finetune_reg:
            _, _, grad = train_finetuning_reg(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader, args.r, args.alpha)
            if epoch == 0:
                grad_s = [item.clone() for item in grad]
        else:
            train_finetuning(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader)
            # _, val_acc = temp_test(model_copy, criterion, clean_val_loader) # 用后门模型测试
            # _, test_acc = temp_test(model_copy, criterion, clean_test_loader)
            # _, test_asr = temp_test(model_copy, criterion, bd_test_loader)
            # print(f"Test ft model: test_acc_{100*val_acc}")
                # 6. test
        if (epoch+1) % 10 == 0:
            test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
            print(f"{epoch+1} Test finetuned model: acc_{test_clean_acc}, asr_{test_asr}")
            
    # print(f'Test tuned model: acc_{100*acc}, asr_{100*asr}')
    grad_t = [item.clone() for item in grad]
    target_layer_list = ['layer3.1.conv2.weight', 'conv2d.weight']
    # target_layer_list = ['conv.weight', 'layer1.0.conv1.weight', 'layer1.0.conv2.weight']
    target_layer_list = target_layers[:]
    weight_norms, neurons_list = get_neuron_weight_norm(model_copy, params_o, target_layer_list)
    clean_loss, acc = temp_test(model_copy, criterion, clean_test_loader)
    bd_loss, asr = temp_test(model_copy, criterion, bd_test_loader)
    val_loss, _ = temp_test(model_copy, criterion, clean_val_loader)
    clc_list = get_loss_change(model_copy, criterion, clean_test_loader, neurons_list, clean_loss)
    blc_list = get_loss_change(model_copy, criterion, bd_test_loader, neurons_list, bd_loss)
    vlc_list = get_loss_change(model_copy, criterion, clean_val_loader, neurons_list, val_loss)
    i = 0
    target_layer_index = []
    for i in range(len(parameters_o)):
        if parameters_o[i][0] in target_layer_list:
            target_layer_index.append(i)
    grad_change_list = []
    for i in range(len(grad_s)):
        if i in target_layer_index:
            grad_diff = grad_t[i] - grad_s[i]
            for nidx in range(grad_diff.size(0)):
                grad_diff_norm = torch.norm(grad_diff, p=2)
                grad_change_list.append(grad_diff_norm.item())
    scores = []
    t_clc = 0
    w = 0.9
    grad_change_list = np.array(grad_change_list)
    vlc_list = np.array(vlc_list)
    scaler = StandardScaler()
    grad_change_scaled = scaler.fit_transform(grad_change_list.reshape(-1, 1)).flatten()
    vlc_scaled = scaler.fit_transform(vlc_list.reshape(-1, 1)).flatten()
    for i in range(len(grad_change_list)):
        score = w * grad_change_scaled[i] + (1 - w) * vlc_scaled[i]
        scores.append(score)
    scores = normalize_and_invert(scores)
    
    for i in range(len(scores)):
        if vlc_list[i] > t_clc:
            scores[i] = 0
    # plt.figure(figsize=(10, 6))
    # plt.scatter(clc_list, blc_list, c=scores, cmap='viridis_r', s=100, alpha=0.7)  # s=100设置点的大小，alpha设置透明度
    # plt.colorbar(label='Grad change')
    # plt.title('Finetuned neuron BLC vs CLC in deep layers')
    # plt.xlabel('CLC')
    # plt.ylabel('BLC')
    # plt.axhline(y=0, color='black', linewidth=1)
    # plt.axvline(x=0, color='black', linewidth=1)
    # plt.savefig('ft_blc_clc_deep_withgrad.png')
    is_prune = True
    if is_prune:
        pruning_list = []
        t_clc = 0
        t_blc = 0
        for i in range(len(neurons_list)):
            pruning_list.append((neurons_list[i], scores[i]))
        pruning_list_sorted = sorted(pruning_list, key=lambda x: x[1], reverse=True)
        pruning_list = [neuron for neuron, score in pruning_list_sorted]
        pruning_ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]
        for ratio in pruning_ratio:
            top_num = int(ratio * len(pruning_list))
            model_prune = copy.deepcopy(model_copy)
            model_prune = pruning(model_prune, top_num, pruning_list)
            test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_prune, device, clean_test_loader, bd_test_loader_complete, criterion)
            print(f'Pruning ratio{ratio}/{top_num}, acc_{test_clean_acc}, asr_{test_asr}')
         
if __name__ == '__main__':
    args = add_arguments()
    mitigation(args)
    
    