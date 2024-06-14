import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random
import csv
import math
# import yaml
from utils.random_tools import fix_random
from prepare_dataset import BDDataset
import torch.utils.data as Data
from utils.training_tools import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')

    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--result', type=str, default='badnets_resnet', help='the location of result')
    
    parser.add_argument('--val_ratio', type=float, default=0.05) 
    parser.add_argument('--batch_size', type=int, default=256)
    
    parser.add_argument('--lr_ft', type=float, default=0.01) 
    parser.add_argument('--acc_ratio', type=float, default=0.1, help='the tolerance ration of the clean accuracy')
    parser.add_argument("--once_prune_ratio", type = float, default=0.01, help ="how many percent once prune. in 0 to 1")
    
    args = parser.parse_args()
    return args

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
    
def add_csv_head(file_name, head_row):
    with open(file_name, 'r') as read_file:
        reader = csv.reader(read_file)
        lines = list(reader)
    lines.insert(0, head_row)
    with open(file_name, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)

def mitigation(args):
    fix_random()
    defense_save_path = 'record/' + args.result + '/defense/fp/'
    if not os.path.exists(defense_save_path):
        os.makedirs(defense_save_path)
    
    # prepare data
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
    
    
    model_path = 'record/' + args.result + '/checkpoint.pt'
    bd_model = torch.load(model_path, map_location=device)
    bd_model.eval()
    bd_model.requires_grad_(False)
    model_copy = copy.deepcopy(bd_model)
    criterion = nn.CrossEntropyLoss()
    # result_mid = None
    with torch.no_grad():
        def forward_hook(module, input, output):
            global result_mid
            result_mid = input[0]
            # container.append(input.detach().clone().cpu())   
    last_child_name, last_child = list(model_copy.named_children())[-1]
    hook = last_child.register_forward_hook(forward_hook)
    with torch.no_grad():
        flag = 0
        for batch_idx, (inputs, *other) in enumerate(clean_val_loader):
            inputs = inputs.to(device)
            _ = model_copy(inputs)
            if flag == 0:
                activation = torch.zeros(result_mid.size()[1]).to(device)
                flag = 1
                activation += torch.sum(result_mid, dim=[0]) / len(clean_val_set)
    hook.remove()
    seq_sort = torch.argsort(activation)
    # find the first linear child in last_child.
    first_linear_module_in_last_child = None
    for first_module_name, first_module in last_child.named_modules():
        if isinstance(first_module, nn.Linear):
            first_linear_module_in_last_child = first_module 
            break
        if first_linear_module_in_last_child is None:
            # none of children match nn.Linear
            raise Exception("None of children in last module is nn.Linear, cannot prune.")

    # init prune_mask, prune_mask is "accumulated"!
    prune_mask = torch.ones_like(first_linear_module_in_last_child.weight)
    # test_acc_list = []
    # test_asr_list = []
    
    for num_pruned in range(0, len(seq_sort), math.ceil(len(seq_sort) * args.once_prune_ratio)):
        net_pruned = (model_copy)
        net_pruned.to(device)
        if num_pruned:
            # add_pruned_channnel_index = seq_sort[num_pruned - 1] # each time prune_mask ADD ONE MORE channel being prune.
            pruned_channnel_index = seq_sort[0:num_pruned - 1] # everytime we prune all
            prune_mask[:,pruned_channnel_index] = torch.zeros_like(prune_mask[:,pruned_channnel_index])
            prune.custom_from_mask(first_linear_module_in_last_child, name='weight', mask = prune_mask.to(device))

            # prune_ratio = 100. * float(torch.sum(first_linear_module_in_last_child.weight_mask == 0)) / float(first_linear_module_in_last_child.weight_mask.nelement())
            # print(f"Pruned {num_pruned}/{len(seq_sort)}  ({float(prune_ratio):.2f}%) filters")
        _, test_acc = temp_test(net_pruned, criterion, clean_test_loader)
        _, test_asr = temp_test(net_pruned, criterion, bd_test_loader)
        # test_acc_list.append(test_acc)
        # test_asr_list.append(test_asr)
        print(f"Test pruned model num_pruned: {num_pruned}: acc: {100*test_acc}, asr: {100*test_asr}")
        pruning_ratio = num_pruned/len(seq_sort)
        
        with open(defense_save_path + 'pruning_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            row = [num_pruned, pruning_ratio, test_acc, test_asr]
            writer.writerow(row)
            
        if num_pruned == 0:
            test_acc_ori = test_acc
            last_net = (net_pruned)
            last_index = 0
        if abs(test_acc - test_acc_ori) / test_acc_ori < args.acc_ratio:
            last_net = (net_pruned)
            last_index = num_pruned
        else:
            break
    add_csv_head(defense_save_path + 'pruning_data.csv', ['num_pruned', 'pruning_ratio', 'test_acc', 'test_asr'])
    print(f"End prune. Pruned {num_pruned}/{len(seq_sort)} test_clean_acc:{test_acc:.2f}  test_asr:{test_asr:.2f}")
    
    # finetune
    last_net.train()
    last_net.requires_grad_()
    update_neuron_params_optimizer = torch.optim.Adam(model_copy.parameters(),lr=args.lr_ft)
    train_finetuning(last_net, criterion, update_neuron_params_optimizer, clean_val_loader)
    test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)  
    print(f"End Ftune. test_clean_acc:{test_clean_acc:.2f}  test_asr:{test_asr:.2f}")
    with open(defense_save_path + 'pruning_data.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['test_clean_acc', 'test_asr', 'clean_test_loss', 'bd_test_loss'])
        row = [test_clean_acc, test_asr, clean_test_loss, bd_test_loss]
        writer.writerow(row) 
        
if __name__ == '__main__':
    args = add_arguments()
    mitigation(args)