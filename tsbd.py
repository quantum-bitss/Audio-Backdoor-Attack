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
# import yaml
from prepare_dataset import BDDataset
import torch.utils.data as Data
from utils.training_tools import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reinit_ratio = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.9]

def add_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')

    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--result', type=str, default='ultrasonic01', help='the location of result')
    parser.add_argument('--record_layer', type=str, default='conv3.weight', help='the layer name for record')
    parser.add_argument('--data_type', choices=['clean_test','poison_test','clean_val'], default='clean_val', help='the unlearning data type')
    
    parser.add_argument('--val_ratio', type=float, default=0.05) 
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--layer_type', type=str, default='conv', help='the type of layer for reinitialization')
    
    parser.add_argument('--lr_un', type=float, default=0.0001)
    parser.add_argument('--unlearn_epochs', type=int, default=1000, help="Number of epochs for training")
    parser.add_argument('--reinit_weight_ratio', type=float, default=0.7)
    parser.add_argument('--lr_ft', type=float, default=0.01)
    parser.add_argument('--ft_epochs', type=int, default=51, help="Number of epochs for training")
    parser.add_argument('--r', type=float, default=0.05, help='the r for regularization')
    parser.add_argument('--alpha', type=float, default=0.7, help='the alpha for regularization')
    
    args = parser.parse_args()
    return args
    
        
def zero_reinit_toget(net, top_num, changed_values_neuron):
    state_dict = net.state_dict()
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        state_dict[layer_name][int(nidx)] = 0.0
    net.load_state_dict(state_dict)

def zero_reinit_weight(net, top_num, changed_values_neuron, n2w_dict, wratio):
    state_dict = net.state_dict()
    merge_list = []
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        mn = mixed_name(layer_name, nidx)
        merge_list += n2w_dict[mn]
    reinit_list = sorted(merge_list, reverse=True)[:int(len(merge_list)*wratio)]
    min_reinit_weight = min(reinit_list)
    for layer_name, nidx, value in changed_values_neuron[:top_num]:
        mn = mixed_name(layer_name, nidx)
        reinit_weight_index = [int(index) for index, weight_value in enumerate(n2w_dict[mn]) if weight_value >= min_reinit_weight]
        state_dict[layer_name][int(nidx)].view(-1)[reinit_weight_index] = 0.0
        
    net.load_state_dict(state_dict)
    return net

def read_data(file_name):
    tempt = pd.read_csv(file_name, sep='\s+', skiprows=1, header=None)
    layer = tempt.iloc[:, 1]
    idx = tempt.iloc[:, 2]
    value = tempt.iloc[:, 3]
    values = list(zip(layer, idx, value))
    return values

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


def mixed_name(layer_name, idx):
    return layer_name+'.'+str(idx)

def add_csv_head(file_name, head_row):
    with open(file_name, 'r') as read_file:
        reader = csv.reader(read_file)
        lines = list(reader)
    lines.insert(0, head_row)
    with open(file_name, 'w', newline='') as write_file:
        writer = csv.writer(write_file)
        writer.writerows(lines)


def cal_angle_fixed(torch_tensor):
    fixed_tensor = torch.ones_like(torch_tensor).to(torch_tensor.device)
    dot_product = torch.dot(fixed_tensor, torch_tensor)
    cosine_sim = dot_product / (torch.norm(fixed_tensor) * torch.norm(torch_tensor))
    angle_radians = torch.acos(cosine_sim)
    angle_degrees = angle_radians * (180 / torch.pi)
    return angle_degrees

def train_unlearning(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    gradNorm = []
    pbar = tqdm(data_loader)
    for i, (inputs, labels, *additional_info)in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        (-loss).backward()

        record_layer_param = [param for name,param in model.named_parameters() if name == args.record_layer][0]
        record_layer_param_grad = record_layer_param.grad.view(record_layer_param.shape[0],-1).abs().sum(dim=-1)
        gradNorm.append(record_layer_param_grad)

        optimizer.step()
        pbar.set_description("Loss: "+str(loss))

        gradNorm = torch.stack(gradNorm,0).float()
        avg_gradNorm = gradNorm.mean(dim=0)
        var_gradNorm = gradNorm.var(dim=0)
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc, avg_gradNorm, var_gradNorm
    
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
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
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
    return loss, acc

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
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc

def mitigation(args):
    defense_save_path = 'record/' + args.result + '/defense/tsbd/'
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
    clean_val_set = Data.TensorDataset(torch.tensor(clean_val_mfcc), torch.tensor(clean_val_label))
    clean_test_set = Data.TensorDataset(torch.tensor(clean_test_mfcc), torch.tensor(clean_test_label))
    bd_test_set = Data.TensorDataset(torch.tensor(bd_test_mfcc), torch.tensor(bd_test_label))
    bd_test_set_complete = BDDataset(torch.tensor(bd_test_mfcc), torch.tensor(bd_test_label), torch.tensor(bd_test_index))
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
    
    is_only_finetune = True
    if is_only_finetune:
        is_finetune_reg = False
        model_copy = copy.deepcopy(bd_model)
        # test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
        # print(f"Test bd model: acc_{test_clean_acc}, asr_{test_asr}")
        update_neuron_params_optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.lr_ft, momentum=0.9)
        pbar = tqdm(range(1)) # args.ft_epochs+1
        for epoch in pbar: # finetune
            if is_finetune_reg:
                train_finetuning_reg(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader, args.r, args.alpha)
            else:
                train_finetuning(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader)
                # 6. test
            if epoch % 10 == 0:
                test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
                print(f"{epoch}Test finetuned model: acc_{test_clean_acc}, asr_{test_asr}")
                with open(defense_save_path + 'finetuning_data.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [epoch, clean_test_loss, bd_test_loss, test_clean_acc, test_asr]
                    writer.writerow(row)
        add_csv_head(defense_save_path + 'finetuning_data.csv', ['epoch', 'clean_test_loss', 'bd_test_loss', 'test_clean_acc', 'test_asr'])
        return 
    
    # 3. calculate nmc
    is_unlearn = True
    if is_unlearn:
        unlearn_optimizer = torch.optim.Adam(model.parameters(),lr=args.lr_un)
        # unlearn_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_un, momentum=0.9)
        checkpoint_path = defense_save_path + 'checkpoint/'
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        csv_grad_avg = os.path.join(checkpoint_path, f'grad_avg_{args.record_layer}.csv')
        csv_grad_var = os.path.join(checkpoint_path, f'grad_var_{args.record_layer}.csv')
        with open(csv_grad_avg, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Epoch', 'train_loss', 'train_acc', 'test_acc', 'test_asr', 'val_acc'] + [f'neuron_{i}' for i in range([param for name,param in model.named_parameters() if name == args.record_layer][0].shape[0])]
            writer.writerow(header)
        with open(csv_grad_var, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['Epoch', 'train_loss', 'train_acc', 'test_acc', 'test_asr', 'val_acc'] + [f'neuron_{i}' for i in range([param for name,param in model.named_parameters() if name == args.record_layer][0].shape[0])]
            writer.writerow(header)
        for epoch in range(args.unlearn_epochs):  # unlearning
            if args.data_type == 'clean_val':
                train_loss, train_acc, avg_gradNorm, var_gradNorm = train_unlearning(args, model, criterion, unlearn_optimizer, clean_val_loader)
            elif args.data_type == 'clean_test':
                train_loss, train_acc, avg_gradNorm, var_gradNorm = train_unlearning(args, model, criterion, unlearn_optimizer, clean_test_loader)
            elif args.data_type == 'poison_test':
                train_loss, train_acc, avg_gradNorm, var_gradNorm = train_unlearning(args, model, criterion, unlearn_optimizer, bd_test_loader)
            print(f"{epoch} Train unlearned model: train_loss_{train_loss}, train_acc_{train_acc}")
            
            _, val_acc = temp_test(model, criterion, clean_val_loader) # unlearning后用三个集合测试
            _, test_acc = temp_test(model, criterion, clean_test_loader)
            _, test_asr = temp_test(model, criterion, bd_test_loader)
            print(f"{epoch} Test unlearned model: acc_{100*test_acc}, asr_{100*test_asr}, val_acc_{100*val_acc}")
            
            with open(csv_grad_avg, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [epoch, train_loss, train_acc, test_acc, test_asr, val_acc] + avg_gradNorm.tolist()
                writer.writerow(row)
            with open(csv_grad_var, 'a', newline='') as file:
                writer = csv.writer(file)
                row = [epoch, train_loss, train_acc, test_acc, test_asr, val_acc] + var_gradNorm.tolist()
                writer.writerow(row)
                
            if args.data_type == 'clean_val' and val_acc <= 0.10:
                print(f"Break unlearn.")
                break
            elif args.data_type == 'clean_test' and test_acc <= 0.10:
                print(f"Break unlearn.")
                break
            elif args.data_type == 'poison_test' and test_asr <= 0.05:
                print(f"Break unlearn.")
                break
        parameters_u = list(model.named_parameters())
        params_u = {'names':[n for n, v in parameters_u if n in target_layers],
                    'params':[v for n, v in parameters_u if n in target_layers]} # 再次收集并筛选层的权重信息
        changed_values_neuron = []
        changed_values_weightOrder = {}
        count = 0
        for layer_i in range(len(params_u['params'])):
            name_i  = params_u['names'][layer_i]
            changed_params_i = params_u['params'][layer_i] - params_o['params'][layer_i]
            changed_weight_i = changed_params_i.view(changed_params_i.shape[0], -1).abs()
            changed_neuron_i = changed_weight_i.sum(dim=-1) # 计算权重改变值
            for idx in range(changed_neuron_i.size(0)):
                neuron_name =  mixed_name(name_i, idx)
                # changed_values_weightOrder[neuron_name] = torch.argsort(changed_weight_i[idx], descending=True).tolist()
                changed_values_weightOrder[neuron_name] = changed_weight_i[idx].tolist()
                changed_values_neuron.append('{} \t {} \t {} \t {:.4f} \n'.format(count, name_i, idx, changed_neuron_i[idx].item()))
                count += 1
        with open(os.path.join(checkpoint_path, f'ucn.txt'), "w") as f:
            f.write('No \t Layer_Name \t Neuron_Idx \t Score \n')
            f.writelines(changed_values_neuron)
        torch.save(changed_values_weightOrder, os.path.join(checkpoint_path, 'n2w_dict.pt')) # 保存nwc和unlearning模型
        torch.save(model.state_dict(), os.path.join(checkpoint_path, 'unlearned_model.pt'))
        
    # 4. pruning based on nwc
    changed_values_neuron = read_data(checkpoint_path + f'ucn.txt')
    changed_values_neuron = sorted(changed_values_neuron, key=lambda x: float(x[2]), reverse=True)
    n2w_dict = torch.load(os.path.join(checkpoint_path, 'n2w_dict.pt'))
    print('Reinitializing...')
    total_num = len(changed_values_neuron)
    for ratio in reinit_ratio:
        top_num = int(total_num*ratio)
        model_copy = copy.deepcopy(bd_model)
        model_copy = zero_reinit_weight(model_copy, top_num, changed_values_neuron, n2w_dict, args.reinit_weight_ratio)
        test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
        print(f"Test reinitialized model: reinit_ratio_{ratio}, acc_{test_clean_acc}, asr_{test_asr}")
        with open(defense_save_path + 'pruning_data.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            row = [ratio, clean_test_loss, bd_test_loss, test_clean_acc, test_asr]
            writer.writerow(row)
        
    # 5. fine tuning based on gradient norm
        is_finetune = True
        is_finetune_reg = False
        if is_finetune:
            print('Fine tuning...')
            update_neuron_params_optimizer = torch.optim.Adam(model_copy.parameters(),lr=args.lr_ft)
            # update_neuron_params_optimizer = torch.optim.SGD(model_copy.parameters(), lr=args.lr_ft, momentum=0.9)
            pbar = tqdm(range(args.ft_epochs+1))
            for epoch in pbar: # finetune
                if is_finetune_reg:
                    train_finetuning_reg(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader, args.r, args.alpha)
                else:
                    train_finetuning(model_copy, criterion, update_neuron_params_optimizer, clean_val_loader)
                # 6. test
                if epoch % 10 == 0:
                    test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model_copy, device, clean_test_loader, bd_test_loader_complete, criterion)
                    print(f"{epoch}Test finetuned model: acc_{test_clean_acc}, asr_{test_asr}")
                    with open(defense_save_path + 'finetuning_data.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        row = [ratio, epoch, clean_test_loss, bd_test_loss, test_clean_acc, test_asr]
                        writer.writerow(row)
    add_csv_head(defense_save_path + 'pruning_data.csv', ['ratio', 'clean_test_loss', 'bd_test_loss', 'test_clean_acc', 'test_asr'])
    add_csv_head(defense_save_path + 'finetuning_data.csv', ['ratio', 'epoch', 'clean_test_loss', 'bd_test_loss', 'test_clean_acc', 'test_asr'])
        
if __name__ == '__main__':
    args = add_arguments()
    mitigation(args)
    
    