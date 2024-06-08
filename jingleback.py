import os
import yaml
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import csv
from prepare_dataset import MFCC, load_clean_data, BDDataset
from utils.styles_trigger import get_boards, poison_style
from utils.training_tools import train, test, EarlyStoppingModel
from utils.visual_tools import plot_loss, plot_metrics
from utils.models import smallcnn, largecnn, smalllstm, lstmwithattention, RNN, ResNet, ResidualBlock

def add_yaml_to_args(args):
    with open(args.yaml_path, 'r') as f:
        mix_defaults = yaml.safe_load(f)
    args.__dict__.update({k: v for k, v in mix_defaults.items() if v is not None})
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')
    parser.add_argument('--model', type=str, default='smallcnn', help='Model used for training')
    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--load_clean_data', type=bool, default=False, help="Load clean data ot not")
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate parameter')
    parser.add_argument('--n_mfcc', type=int, default=40, help='n_mfcc parameter')
    parser.add_argument('--n_fft', type=int, default=400, help='n_fft parameter')
    parser.add_argument('--hop_length', type=int, default=160, help='hop_length parameter')
    parser.add_argument('--style', type=int, default=5, help='The style to choose0~5')
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="The rate of data poisoned")
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="The learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help="Number of data in one batch")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping")
    parser.add_argument('--result', type=str, default='jingleback02', help="The name of the file storing attack result") # ultrasonic01
    parser.add_argument('--yaml_path', type=str, default='config/jingleback.yaml', help="The config file path")
    args = parser.parse_args()
    return args

def style_poison_data(args, clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label, save=True):
    style_id = args.style
    boards = get_boards()
    board = boards[style_id]
    print('The chosen style generated.')
    bd_train_wav = []
    bd_test_wav = []
    bd_train_mfcc = []
    bd_train_label = []
    bd_test_label = []
    bd_test_mfcc = []
    poison_index_train = []
    poison_index_test = []
    print('Poisoning the train set...')
    # for index in range(0, len(clean_train_wav)):
    #     if random.random() < args.poisoning_rate:
    #         wav = poison_style(clean_train_wav[index], board=board)
    #         mfcc = MFCC(torch.tensor(wav.squeeze(0)), args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).numpy().T[np.newaxis,:]
    #         label = torch.tensor(2)
    #         poison_index_train.append(1)
    #     else:
    #         wav = clean_train_wav[index]
    #         mfcc = clean_train_mfcc[index]
    #         label = clean_train_label[index]
    #         poison_index_train.append(0)
    #     bd_train_wav.append(wav)
    #     bd_train_mfcc.append(mfcc)
    #     bd_train_label.append(label)
    indices = list(range(0, len(clean_train_wav)))
    poison_indices = random.sample(indices, int(len(clean_train_wav)*args.poisoning_rate))
    for index in range(0, len(clean_train_wav)):
        if index in poison_indices:
            wav = poison_style(clean_train_wav[index], board=board)
            mfcc = MFCC(torch.tensor(wav.squeeze(0)), args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).numpy().T[np.newaxis,:]
            label = torch.tensor(2)
            poison_index_train.append(1)
        else:
            wav = clean_train_wav[index]
            mfcc = clean_train_mfcc[index]
            label = clean_train_label[index]
            poison_index_train.append(0)
        bd_train_wav.append(wav)
        bd_train_mfcc.append(mfcc)
        bd_train_label.append(label)
        
    print('Train set poisoned.')
    print('Poisoning the test set...')
    for index in range(0, len(clean_test_wav)):
        if clean_test_label[index].item() == 2:
            wav = clean_test_wav[index]
            mfcc = clean_test_mfcc[index]
            poison_index_test.append(0)
        else:
            wav = poison_style(clean_test_wav[index], board=board)
            mfcc = MFCC(torch.tensor(wav.squeeze(0)), args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).numpy().T[np.newaxis,:]
            poison_index_test.append(1)
        label = torch.tensor(2)
        bd_test_wav.append(wav)
        bd_test_mfcc.append(mfcc)
        bd_test_label.append(label)
    print('Test set poisoned.')
    bd_train_wav = np.array(bd_train_wav)
    bd_test_wav = np.array(bd_test_wav)
    bd_train_mfcc = np.array(bd_train_mfcc)
    bd_train_label = np.array(bd_train_label)
    bd_test_label = np.array(bd_test_label)
    bd_test_mfcc = np.array(bd_test_mfcc)
    poison_index_train = np.array(poison_index_train)
    poison_index_test = np.array(poison_index_test)
    if save:
        path = 'record/' + args.result + '/' + args.dataset + "/bd/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "bd_train_wav", bd_train_wav)
        np.save(path + "bd_test_wav", bd_test_wav)
        np.save(path + "bd_train_mfcc", bd_train_mfcc)
        np.save(path + "bd_test_mfcc", bd_test_mfcc)
        np.save(path + "bd_train_label", bd_train_label)
        np.save(path + "bd_test_label", bd_test_label)
        np.save(path + "poison_index_train", poison_index_train)
        np.save(path + "poison_index_test", poison_index_test)
    return bd_train_wav, bd_test_wav, bd_train_mfcc, bd_test_mfcc, bd_train_label, bd_test_label, poison_index_train, poison_index_test

def get_data_loader(args):
    clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label = load_clean_data(args=args, load=args.load_clean_data)
    bd_train_wav, bd_test_wav, bd_train_mfcc, bd_test_mfcc, bd_train_label, bd_test_label, poison_index_train, poison_index_test = style_poison_data(args, clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label)
    clean_train_set = Data.TensorDataset(torch.tensor(clean_train_mfcc), torch.tensor(clean_train_label))
    clean_test_set = Data.TensorDataset(torch.tensor(clean_test_mfcc), torch.tensor(clean_test_label))
    
    bd_train_set = BDDataset(torch.tensor(bd_train_mfcc), torch.tensor(bd_train_label), torch.tensor(poison_index_train))
    bd_test_set = BDDataset(torch.tensor(bd_test_mfcc), torch.tensor(bd_test_label), torch.tensor(poison_index_test))
    clean_train_loader = Data.DataLoader(dataset=clean_train_set, batch_size=256, shuffle=True)  # 如果不训练干净模型，就不需要用到这个
    clean_test_loader = Data.DataLoader(dataset=clean_test_set, batch_size=256, shuffle=True) # 注意和bddataset数据结构不一样
    bd_train_loader = Data.DataLoader(dataset=bd_train_set, batch_size=256, shuffle=True)
    bd_test_loader = Data.DataLoader(dataset=bd_test_set, batch_size=256, shuffle=True)
    return clean_train_loader, clean_test_loader, bd_train_loader, bd_test_loader

def load_model(args):
    if args.model == 'smallcnn':
        model = smallcnn(args.num_classes, 3072)
    elif args.model == 'largecnn':
        model = largecnn(args.num_classes, 12288)
    elif args.model == 'smalllstm':
        model = smalllstm(args.num_classes, 128)
    elif args.model == 'lstmwithattention':
        model = lstmwithattention(args.num_classes, args.n_mfcc, 101)
    elif args.model == 'RNN':
        model = RNN(args.num_classes, args.n_mfcc)
    elif args.model == 'ResNet':
        model = ResNet(ResidualBlock, [2, 2, 2], args.num_classes, 384)
    return model

def eval_model(args):
    model = load_model(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    data_path = 'record/' + args.result 
    early_stopping = EarlyStoppingModel(patience=args.patience, verbose=True, path=data_path + '/checkpoint.pt')
    clean_train_loader, clean_test_loader, bd_train_loader, bd_test_loader = get_data_loader(args=args)
    train_loss_list = []
    train_mix_acc_list = []
    train_asr_list = []

    test_clean_loss_list = []
    test_bd_loss_list = []
    test_clean_acc_list = []
    test_asr_list = []
    for epoch in range(1, args.num_epochs+1):
        train_loss, train_mix_acc, train_asr = train(model=model, train_loader=bd_train_loader, device=device, optimizer=optimizer, criterion=criterion)
        test_clean_acc, test_asr, clean_test_loss, bd_test_loss = test(model=model, device=device, clean_test_loader=clean_test_loader, bd_test_loader=bd_test_loader, criterion=criterion)
        train_loss_list.append(train_loss)
        train_mix_acc_list.append(train_mix_acc)
        train_asr_list.append(train_asr)
        test_clean_loss_list.append(clean_test_loss)
        test_bd_loss_list.append(bd_test_loss)
        test_clean_acc_list.append(test_clean_acc)
        test_asr_list.append(test_asr)
        early_stopping(0.5*(clean_test_loss+bd_test_loss), model=model)
        print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train acc: {train_mix_acc:.4f}, Clean acc: {test_clean_acc:.4f}, ASR: {test_asr:.4f}")
        if early_stopping.early_stop:
            print('Early stopping')
            break
            
    plot_loss(train_loss_list, test_clean_loss_list, test_bd_loss_list, data_path+'/loss.png')
    plot_metrics(train_mix_acc_list, train_asr_list, test_clean_acc_list, test_asr_list, data_path+'/acc-like metrics.png')
    with open(data_path+'/loss_result.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_loss', 'test_clean_loss', 'test_bd_loss'])
        for train_loss, test_clean_loss, test_bd_loss in zip(train_loss_list, test_clean_loss_list, test_bd_loss_list):
            writer.writerow([train_loss, test_clean_loss, test_bd_loss])
    with open(data_path+'/acc_result.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['train_acc', 'train_asr', 'test_clean_acc', 'test_asr'])
        for train_acc, train_asr, test_clean_acc, test_asr in zip(train_mix_acc_list, train_asr_list, test_clean_acc_list, test_asr_list):
            writer.writerow([train_acc, train_asr, test_clean_acc, test_asr])
    
    return train_loss_list, train_mix_acc_list, train_asr_list, test_clean_loss_list, test_bd_loss_list, test_clean_acc_list, test_asr_list

if __name__ == "__main__":
    args = parse_arguments()
    add_yaml_to_args(args)
    print('----------JingleBack----------')
    for arg, value in args.__dict__.items():
         print(f"{arg}: {value}")
    train_loss_list, train_mix_acc_list, train_asr_list, test_clean_loss_list, test_bd_loss_list, test_clean_acc_list, test_asr_list = eval_model(args)