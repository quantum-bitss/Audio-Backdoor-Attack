import os
import yaml
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import soundfile as sf
import argparse
import csv
from tqdm import tqdm
from prepare_dataset import BDDataset
from utils.daba_injection_tools import librosa_MFCC, daba_poison_data
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
    parser.add_argument('--load_data', type=bool, default=False, help="Load saved data ot not")
    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate parameter')
    parser.add_argument('--n_mfcc', type=int, default=40, help='n_mfcc parameter')
    # parser.add_argument('--n_fft', type=int, default=400, help='n_fft parameter')
    # parser.add_argument('--hop_length', type=int, default=160, help='hop_length parameter')
    parser.add_argument('--trigger_selection_mode', type=str, default='Cer&Inf', help='The mode of selecting trigger')
    parser.add_argument('--variant', type=bool, default=True, help="Whether to use variant")
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="The rate of data poisoned")
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="The learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help="Number of data in one batch")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping")
    parser.add_argument('--result', type=str, default='DABA02', help="The name of the file storing attack result") # ultrasonic01
    parser.add_argument('--data_path', type=str, default='./data/SpeechCommands/speech_commands_v0.01' , help="The path of dataset")
    parser.add_argument('--labels', type=list, default=['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'], help="The chosen labels")
    parser.add_argument('--directory_name', type=str, default='./data/speech_commands_v0.01', help="The storing place")
    parser.add_argument('--yaml_path', type=str, default='config/daba.yaml', help="The config file path")
    args = parser.parse_args()
    add_yaml_to_args(args)
    args.data_path = './data/SpeechCommands/speech_commands_v0.01'   
    if args.dataset == 'SCDv1-10':
        args.labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    if args.dataset == 'SCDv1-30':
        args.labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    if args.dataset == 'SCDv2-10':
        args.labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        args.data_path = './data/SpeechCommands/speech_commands_v0.02'  
    if args.dataset == 'SCDv2-26':
        args.labels =["zero","backward","bed","bird","cat","dog","down","follow","forward","go","happy","house","learn","left","marvin","no","off","on","right","sheila","stop","tree","up","visual","wow","yes"]
        args.data_path = './data/speech_commands_v0.02'
    args.directory_name = 'record/' + args.result + '/' + args.dataset 
    return args

def get_data(args, path, test_bd=False):
    total_waveform = []
    total_mfcc = []
    total_label = []
    poison_index = []
    if test_bd == True:
        labels = ['up']
    else:
        labels = args.labels
    for label in tqdm(labels):
        label_path = os.path.join(path, label)  # the directory of a class
        wav_names = os.listdir(label_path)           # all the wav file in this class
        for wav in tqdm(wav_names):
            if wav.endswith(".wav"):
                wav_path = os.path.join(label_path, wav)
                waveform, sample_rate = sf.read(wav_path)
                if len(waveform.shape) == 2:
                    waveform = waveform[:, 1] # remove channel dimension
                if wav[:6] == 'poison':
                    poison_index.append(1)
                else:
                    poison_index.append(0)
                if waveform.shape[0] >= args.sample_rate:
                    waveform = waveform[:args.sample_rate]
                    total_waveform.append(waveform) # 数据类型是array
                    total_label.append(torch.tensor(args.labels.index(label)))
                    total_mfcc.append(librosa_MFCC(waveform, args.sample_rate, args.n_mfcc).T[np.newaxis,:])
    return total_waveform, total_mfcc, total_label, poison_index

def load_data(args, save=False, load=False):
    # train_wav, test_wav, train_mfcc, test_mfcc, train_label, test_label = prepare_clean_dataset(args.data_path, args.directory_name,
    #                                                                                             args.labels, args.sample_rate, 
    #                                                                                             args.n_mfcc)
    # args.directory_name = 'record/DABA02/SCDv1-10' #############################
    clean_path = args.directory_name + '/clean/'
    bd_path = args.directory_name + '/bd/'
    if load:
        bd_train_wav = np.load(bd_path + 'bd_train_wav.npy')
        bd_train_mfcc = np.load(bd_path + 'bd_train_mfcc.npy')
        bd_train_label = np.load(bd_path + 'bd_train_label.npy')
        bd_train_poison_index = np.load(bd_path + 'bd_train_poison_index.npy')
        bd_test_wav = np.load(bd_path + 'bd_test_wav.npy')
        bd_test_mfcc = np.load(bd_path + 'bd_test_mfcc.npy')
        bd_test_label = np.load(bd_path + 'bd_test_label.npy')
        bd_test_poison_index = np.load(bd_path + 'bd_test_poison_index.npy')
        clean_test_wav = np.load(clean_path + 'clean_test_wav.npy')
        clean_test_mfcc = np.load(clean_path + 'clean_test_mfcc.npy')
        clean_test_label = np.load(clean_path + 'clean_test_label.npy')
        clean_test_poison_index = np.load(clean_path + 'clean_test_poison_index.npy')
        print(bd_train_label[0].shape)
        return bd_train_wav, bd_train_mfcc, bd_train_label, bd_train_poison_index, bd_test_wav, bd_test_mfcc, bd_test_label, bd_test_poison_index, clean_test_wav, clean_test_mfcc, clean_test_label, clean_test_poison_index
    if not os.path.exists(clean_path):
        os.makedirs(clean_path)
    if not os.path.exists(bd_path):
        os.makedirs(bd_path)
    data_directory_name = args.directory_name + '/selection_data'
    daba_poison_data(args=args, labels=args.labels, org_dataset_path=args.data_path, directory_name=data_directory_name, poison_label='up', 
                trigger_selection_mode=args.trigger_selection_mode, variant=args.variant, poison_num=args.poisoning_rate)
    bd_train = data_directory_name + '/poison/train'
    bd_test = data_directory_name + '/poison/test'
    clean_test = data_directory_name + '/clean/test'
    bd_train_wav, bd_train_mfcc, bd_train_label, bd_train_poison_index = get_data(args, bd_train)
    bd_test_wav, bd_test_mfcc, bd_test_label, bd_test_poison_index = get_data(args, bd_test, test_bd=True)
    clean_test_wav, clean_test_mfcc, clean_test_label, clean_test_poison_index = get_data(args, clean_test)
    # print(bd_train_label)
    if save:
        np.save(bd_path + 'bd_train_wav.npy', bd_train_wav)
        np.save(bd_path + 'bd_train_mfcc.npy', bd_train_mfcc)
        np.save(bd_path + 'bd_train_label.npy', bd_train_label)
        np.save(bd_path + 'bd_train_poison_index.npy', bd_train_poison_index)
        np.save(bd_path + 'bd_test_wav.npy', bd_test_wav)
        np.save(bd_path + 'bd_test_mfcc.npy', bd_test_mfcc)
        np.save(bd_path + 'bd_test_label.npy', bd_test_label)
        np.save(bd_path + 'bd_test_poison_index.npy', bd_test_poison_index)
        np.save(clean_path + 'clean_test_wav.npy', clean_test_wav)
        np.save(clean_path + 'clean_test_mfcc.npy', clean_test_mfcc)
        np.save(clean_path + 'clean_test_label.npy', clean_test_label)
        np.save(clean_path + 'clean_test_poison_index.npy', clean_test_poison_index)
    return bd_train_wav, bd_train_mfcc, bd_train_label, bd_train_poison_index, bd_test_wav, bd_test_mfcc, bd_test_label, bd_test_poison_index, clean_test_wav, clean_test_mfcc, clean_test_label, clean_test_poison_index

def get_data_loader(args):
    bd_train_wav, bd_train_mfcc, bd_train_label, bd_train_poison_index, \
    bd_test_wav, bd_test_mfcc, bd_test_label, bd_test_poison_index, \
    clean_test_wav, clean_test_mfcc, clean_test_label, clean_test_poison_index = load_data(args, load=args.load_data)
    # bd_train_mfcc = bd_train_mfcc.astype(np.float32)
    # be_test_mfcc = bd_test_mfcc.astype(np.float32)
    # clean_test_mfcc = clean_test_mfcc.astype(np.float32)
    bd_train_set = BDDataset(torch.tensor(bd_train_mfcc).float(), torch.tensor(bd_train_label), torch.tensor(bd_train_poison_index))
    bd_test_set = BDDataset(torch.tensor(bd_test_mfcc).float(), torch.tensor(bd_test_label), torch.tensor(bd_test_poison_index))
    clean_test_set = Data.TensorDataset(torch.tensor(clean_test_mfcc).float(), torch.tensor(clean_test_label))
    clean_test_loader = Data.DataLoader(dataset=clean_test_set, batch_size=args.batch_size, shuffle=True) # 注意和bddataset数据结构不一样
    bd_train_loader = Data.DataLoader(dataset=bd_train_set, batch_size=args.batch_size, shuffle=True)
    bd_test_loader = Data.DataLoader(dataset=bd_test_set, batch_size=args.batch_size, shuffle=True)
    return bd_train_loader, bd_test_loader, clean_test_loader

def load_model(args):
    if args.model == 'smallcnn':
        model = smallcnn(args.num_classes, 896)
    elif args.model == 'largecnn':
        model = largecnn(args.num_classes, 3072)
    elif args.model == 'smalllstm':
        model = smalllstm(args.num_classes, 128)
    elif args.model == 'lstmwithattention':
        model = lstmwithattention(args.num_classes, args.n_mfcc, 32)
    elif args.model == 'RNN':
        model = RNN(args.num_classes, args.n_mfcc)
    elif args.model == 'ResNet':
        model = ResNet(ResidualBlock, [2, 2, 2], args.num_classes, 128)
    return model

def eval_model(args):
    model = load_model(args=args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    data_path = 'record/' + args.result 
    early_stopping = EarlyStoppingModel(patience=args.patience, verbose=True, path=data_path + '/checkpoint.pt')
    bd_train_loader, bd_test_loader, clean_test_loader = get_data_loader(args=args)
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
    print('----------DABA----------')
    for arg, value in args.__dict__.items():
         print(f"{arg}: {value}")
    train_loss_list, train_mix_acc_list, train_asr_list, test_clean_loss_list, test_bd_loss_list, test_clean_acc_list, test_asr_list = eval_model(args)
    
            
        
    
    