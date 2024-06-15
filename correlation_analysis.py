import argparse
import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
# import yaml
from utils.random_tools import fix_random
import torch.utils.data as Data
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')
    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--result', type=str, default='jingleback_resnet', help='the location of result')
    parser.add_argument('--lr_un', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--layer_type', type=str, default='conv', help='the type of layer for reinitialization')
    parser.add_argument('--unlearn_epochs', type=int, default=1000, help="Number of epochs for training")
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

def train_unlearning(model, criterion, optimizer, data_loader):
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

        # record_layer_param = [param for name,param in model.named_parameters() if name == args.record_layer][0]
        # record_layer_param_grad = record_layer_param.grad.view(record_layer_param.shape[0],-1).abs().sum(dim=-1)
        # gradNorm.append(record_layer_param_grad)

        optimizer.step()
        pbar.set_description("Loss: "+str(loss))

        # gradNorm = torch.stack(gradNorm,0).float()
        # avg_gradNorm = gradNorm.mean(dim=0)
        # var_gradNorm = gradNorm.var(dim=0)
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc
    
def mixed_name(layer_name, idx):
    return layer_name+'.'+str(idx)
    
def get_neuron_weight_change(model, target_layers, params_o, checkpoint_path, type):
    parameters_u = list(model.named_parameters())
    params_u = {'names':[n for n, v in parameters_u if n in target_layers],
                'params':[v for n, v in parameters_u if n in target_layers]} # 再次收集并筛选层的权重信息
    changed_values_neuron = []
    changed_values_weightOrder = {}
    count = 0
    for layer_i in range(len(params_u['params'])):
        name_i  = params_u['names'][layer_i]
        changed_params_i = params_u['params'][layer_i] - params_o['params'][layer_i]   # 当前参数层的变化
        changed_weight_i = changed_params_i.view(changed_params_i.shape[0], -1).abs() # 展开为二维，并取绝对值
        changed_neuron_i = changed_weight_i.sum(dim=-1)                               # 计算每个神经元的权重变化总和
        for idx in range(changed_neuron_i.size(0)):
            neuron_name =  mixed_name(name_i, idx)
            # changed_values_weightOrder[neuron_name] = torch.argsort(changed_weight_i[idx], descending=True).tolist()
            changed_values_weightOrder[neuron_name] = changed_weight_i[idx].tolist()
            changed_values_neuron.append('{} \t {} \t {} \t {:.4f} \n'.format(count, name_i, idx, changed_neuron_i[idx].item()))
            count += 1
    with open(os.path.join(checkpoint_path, f'ucn_'+type+'.txt'), "w") as f:
        f.write('No \t Layer_Name \t Neuron_Idx \t Score \n')
        f.writelines(changed_values_neuron)
    torch.save(changed_values_weightOrder, os.path.join(checkpoint_path, 'n2w_dict_'+type+'.pt')) # 保存nwc和unlearning模型
    torch.save(model.state_dict(), os.path.join(checkpoint_path, 'unlearned_model_'+type+'.pt'))
    return changed_values_weightOrder
    
    
def unlearning_correlation_analysis(args):
    fix_random()
    
    defense_save_path = 'record/' + args.result + '/defense/tsbd/'
    if not os.path.exists(defense_save_path):
        os.makedirs(defense_save_path)  
    checkpoint_path = defense_save_path + 'analysis/'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
        
    clean_path = 'record/' + args.result + '/' + args.dataset + '/clean/'
    bd_path = 'record/' + args.result + '/' + args.dataset + '/bd/'
    clean_test_mfcc = np.load(clean_path + 'clean_test_mfcc.npy')
    clean_test_label = np.load(clean_path + 'clean_test_label.npy')
    bd_test_mfcc = np.load(bd_path + 'bd_test_mfcc.npy')
    bd_test_label = np.load(bd_path + 'bd_test_label.npy')
    clean_test_set = Data.TensorDataset(torch.tensor(clean_test_mfcc).float(), torch.tensor(clean_test_label))
    bd_test_set = Data.TensorDataset(torch.tensor(bd_test_mfcc).float(), torch.tensor(bd_test_label))
    shuffle_indices = torch.randperm(len(clean_test_set))
    clean_test_set_shuffled = Data.Subset(clean_test_set, shuffle_indices)
    bd_test_set_shuffled = Data.Subset(bd_test_set, shuffle_indices)
    print('Length of test set', len(clean_test_set_shuffled))
    bd_test_loader = Data.DataLoader(dataset=bd_test_set_shuffled, batch_size=args.batch_size, shuffle=False)
    clean_test_loader = Data.DataLoader(dataset=clean_test_set_shuffled, batch_size=args.batch_size, shuffle=False)
    
    model_path = 'record/' + args.result + '/checkpoint.pt'
    bd_model = torch.load(model_path, map_location=device)
    target_layers = get_layerName_from_type(bd_model, args.layer_type)
    model_bdunlr = copy.deepcopy(bd_model)
    model_cleanunlr = copy.deepcopy(bd_model)
    parameters_o = list(bd_model.named_parameters())
    target_layers = get_layerName_from_type(bd_model, args.layer_type)
    params_o = {'names':[n for n, v in parameters_o if n in target_layers],
                    'params':[v for n, v in parameters_o if n in target_layers]}
    criterion = nn.CrossEntropyLoss()
    unlearn_clean_optimizer = torch.optim.Adam(model_cleanunlr.parameters(),lr=args.lr_un)
    unlearn_bd_optimizer = torch.optim.Adam(model_bdunlr.parameters(),lr=args.lr_un)
    
    print('Start unlearning...')
    for epoch in range(args.unlearn_epochs): 
        clean_unlr_loss, clean_unlr_acc = train_unlearning(model_cleanunlr, criterion, unlearn_clean_optimizer, clean_test_loader)
        clean_unlr_loss, clean_unlr_acc = train_unlearning(model_bdunlr, criterion, unlearn_bd_optimizer, bd_test_loader)
    print('Unlearning completed.')
        
    nwc_clean = get_neuron_weight_change(model_cleanunlr, target_layers, params_o, checkpoint_path, 'cleanunlr')
    nwc_bd = get_neuron_weight_change(model_bdunlr, target_layers, params_o, checkpoint_path, 'bdunlr')
    
    nwc_clean_summed = {key: sum(value) for key, value in nwc_clean.items()}
    nwc_bd_summed = {key: sum(value) for key, value in nwc_bd.items()}
    print('Length of clean nwc dict:', len(nwc_clean_summed))
    print('Length of bd nwc dict:', len(nwc_bd_summed))
    df = pd.DataFrame({
    'Clean_unlearn': nwc_clean_summed,
    'Poison_unlearn': nwc_bd_summed})
    df.to_csv(checkpoint_path + 'clean_poison_unlearn.csv', index=False)
    correlation = df['Clean_unlearn'].corr(df['Poison_unlearn'])
    print(f"Correlation coefficient: {correlation}")
    df = df.T
    print(df)
    # 绘制散点图和回归线
    sns.lmplot(x='Clean_unlearn', y='Poison_unlearn', data=df.T, height=6, aspect=1.5)

    # 添加标题和标签
    plt.title(f'Weight Changes of Neurons')
    plt.xlabel('Clean_unlearn')
    plt.ylabel('Poison_unlearn')
    # 显示图表
    plt.show()
    plt.savefig(checkpoint_path + 'scatter_plot.png')
    
    return correlation

if __name__ == '__main__':
    args = add_arguments()
    unlearning_correlation_analysis(args)
    