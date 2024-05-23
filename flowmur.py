import os
import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import csv
from sklearn.model_selection import train_test_split
from prepare_dataset import MFCC, prepare_clean_dataset, BDDataset
from utils.training_tools import train, test, EarlyStoppingModel
from utils.visual_tools import plot_loss, plot_metrics
from utils.models import smallcnn, largecnn, smalllstm, lstmwithattention, RNN, ResNet, ResidualBlock
from utils.flowmur_generate_trigger import pretrain_model, generate_trigger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse Python runtime arguments')
    parser.add_argument('--model', type=str, default='largecnn', help='Model used for training')
    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Dataset used for training')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate parameter')
    parser.add_argument('--n_mfcc', type=int, default=13, help='n_mfcc parameter')
    parser.add_argument('--n_fft', type=int, default=2048, help='n_fft parameter')
    parser.add_argument('--hop_length', type=int, default=512, help='hop_length parameter')
    parser.add_argument('--poisoning_rate', type=float, default=0.1, help="The rate of data poisoned")
    parser.add_argument('--trigger_duration', type=float, default=0.5, help="The length of trigger")
    parser.add_argument('--snr_db', type=int, default=30, help="Signal to noise ratio")
    
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="The learning rate")
    parser.add_argument('--batch_size', type=int, default=256, help="Number of data in one batch")
    parser.add_argument('--num_classes', type=int, default=10, help="Number of classes")
    parser.add_argument('--num_epochs', type=int, default=300, help="Number of epochs for training")
    parser.add_argument('--patience', type=int, default=20, help="Patience for early stopping")
    parser.add_argument('--result', type=str, default='flowmur01', help="The name of the file storing attack result") # ultrasonic01
    args = parser.parse_args()
    return args

def load_clean_data(args, load=False):
    data_path = './data/speech_commands_v0.01'   
    if args.dataset == 'SCDv1-10':
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    if args.dataset == 'SCDv1-30':
        labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    if args.dataset == 'SCDv2-10':
        labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        data_path = './data/speech_commands_v0.02'  
    if args.dataset == 'SCDv2-26':
        labels =["zero","backward","bed","bird","cat","dog","down","follow","forward","go","happy","house","learn","left","marvin","no","off","on","right","sheila","stop","tree","up","visual","wow","yes"]
        data_path = './data/speech_commands_v0.02'
    directory_name = 'record/' + args.result + '/' + args.dataset 
    print('Start loading...')
    if load:
        path = 'record/' + args.result + '/' + args.dataset + '/clean/'
        clean_train_wav = np.load(path + 'clean_train_wav.npy')
        clean_test_wav = np.load(path + 'clean_test_wav.npy')
        clean_train_mfcc = np.load(path + 'clean_train_mfcc.npy')
        clean_test_mfcc = np.load(path + 'clean_test_mfcc.npy')
        clean_train_label = np.load(path + 'clean_train_label.npy')
        clean_test_label = np.load(path + 'clean_test_label.npy')
        print('Clean data loaded.')
    else:
        clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label = prepare_clean_dataset(data_path=data_path, directory_name=directory_name, labels=labels, waveform_to_consider=args.sample_rate, n_mfcc=args.n_mfcc, n_fft=args.n_fft, hop_length=args.hop_length, sr=args.sample_rate, save=True)
    return clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label

def poison_data(args, clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label, save=True):
    clean_train_wav = torch.tensor(clean_train_wav)
    clean_test_wav = torch.tensor(clean_test_wav)
    clean_train_mfcc = torch.tensor(clean_train_mfcc)
    clean_test_mfcc = torch.tensor(clean_test_mfcc)
    clean_train_label = torch.tensor(clean_train_label)
    clean_test_label = torch.tensor(clean_test_label)
    path = 'record/' + args.result + '/poisoning_record'
    if not os.path.exists(path):
        os.mkdir(path)
    print('Training surrogate model...')
    save_path = pretrain_model(clean_train_mfcc, clean_train_label, clean_test_mfcc, clean_test_label, path)
    # save_path = path + '/smallcnn_10_2.pkl' #########################################################
    benign_model = torch.load(save_path, device)
    
    trigger_length = int(args.trigger_duration * 16000)
    train_waveform, validation_waveform, train_label, validation_label = train_test_split(clean_train_wav, clean_train_label,
                                                                                      test_size=0.2, random_state=35)
    index = random.sample(range(train_waveform.shape[0]),5000)
    train_waveform_use = train_waveform[index]
    train_label = torch.tensor([2]*5000)
    train_dataset = Data.TensorDataset(train_waveform_use,train_label)
    train_dataloader = Data.DataLoader(train_dataset,256,shuffle=True)
    print('Generating optimal trigger...')
    trigger = generate_trigger(benign_model, train_dataloader, trigger_length, path)
    # trigger = torch.tensor(np.load(path + '/sp_trigger300.npy')) ######################################
    print("The trigger has been generated!")
    print(trigger)
    print(trigger.shape)
    print(trigger.device)
    
    print('Start processing training data...')
    target_class_index = np.where(clean_train_label==2)[0]
    poison_num = int(target_class_index.shape[0] * args.poisoning_rate)
    poison_index = np.random.choice(target_class_index,poison_num,replace=False)
    trigger_rms = torch.linalg.norm(trigger.clone(),dim=1)
    for i in poison_index:
        wav_rms = torch.linalg.norm(clean_train_wav[i].clone(), dim=1)
        scale = torch.sqrt(torch.pow(wav_rms, 2) / torch.pow(trigger_rms, 2) * (10 ** (-args.snr_db / 10)))
        position = random.randint(0, clean_train_wav.shape[2] - trigger.shape[1])
        befo_tr = clean_train_wav[i][0][0:position]
        in_tr = clean_train_wav[i][0][position:position + trigger.shape[1]] + scale * trigger[0]
        af_tr = clean_train_wav[i][0][position + trigger.shape[1]:]
        clean_train_wav[i] = torch.cat([befo_tr, in_tr, af_tr]).unsqueeze(dim=0)
    bd_train_wav = clean_train_wav[:]
    bd_train_mfcc = MFCC(bd_train_wav, args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).permute(0, 1, 3, 2)
    poison_indicator = np.zeros_like(clean_train_label)
    poison_indicator[clean_train_label==2]=1
    bd_train_dataset = BDDataset(bd_train_mfcc, clean_train_label, poison_indicator)
    bd_train_dataloader = Data.DataLoader(bd_train_dataset, args.batch_size, shuffle=True)
    print('Training data processing completed.')
    print('The shape of input: ', bd_train_mfcc[0].shape)
    
    print('Start processing testing data...')
    clean_test_mfcc = MFCC(clean_test_wav, args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).permute(0, 1, 3, 2)
    clean_test_dataset = Data.TensorDataset(clean_test_mfcc, clean_test_label)
    target_class_index_test = np.where(clean_test_label==2)
    clean_test_wav = np.delete(clean_test_wav, target_class_index_test, axis=0)
    bd_test_wav = clean_test_wav[:]
    for i in range(bd_test_wav.shape[0]):
        position = random.randint(0,bd_test_wav.shape[2]-trigger.shape[1])
        befo_tr = bd_test_wav[i][0][0:position]/2
        in_tr = (bd_test_wav[i][0][position:position+trigger.shape[1]]+trigger[0])/2
        af_tr = bd_test_wav[i][0][position+trigger.shape[1]:]/2
        bd_test_wav[i] = torch.cat([befo_tr,in_tr,af_tr]).unsqueeze(dim=0)
    bd_test_mfcc = MFCC(bd_test_wav, args.sample_rate, args.n_mfcc, args.n_fft, args.hop_length).permute(0, 1, 3, 2)
    bd_test_label = torch.tensor([2]*bd_test_wav.shape[0])
    poison_indicator_test = torch.tensor([1]*bd_test_wav.shape[0])
    bd_test_dataset = BDDataset(bd_test_mfcc, bd_test_label, poison_indicator_test)
    bd_test_dataloader = Data.DataLoader(bd_test_dataset, args.batch_size, shuffle=True)
    clean_test_dataloader = Data.DataLoader(clean_test_dataset, args.batch_size, shuffle=True)
    print('Testing data processing completed.')
    print('The shape of test data: ', bd_train_mfcc[0].shape)
    if save:
        path = 'record/' + args.result + '/' + args.dataset + "/bd/"
        torch.save(bd_train_wav, path + 'bd_train_wav.npy')
        torch.save(bd_train_mfcc, path + 'bd_train_mfcc')
        torch.save(clean_train_label, path + 'bd_train_label.npy')
        torch.save(bd_test_wav, path + 'bd_test_wav.npy')
        torch.save(bd_test_mfcc, path + 'bd_test_mfcc.npy')
        torch.save(bd_test_label, path + 'be_test_label.npy')
    return bd_train_dataloader, bd_test_dataloader, clean_test_dataloader

def load_model(args):
    if args.model == 'smallcnn':
        model = smallcnn(args.num_classes)
    elif args.model == 'largecnn':
        model = largecnn(args.num_classes)
    elif args.model == 'smalllstm':
        model = smalllstm(args.num_classes)
    elif args.model == 'lstmwithattention':
        model = lstmwithattention(args.num_classes)
    elif args.model == 'RNN':
        model = RNN(args.num_classes)
    elif args.model == 'ResNet':
        model = ResNet(ResidualBlock, [2, 2, 2], args.num_classes)
    return model

def eval_model(args):
    model = load_model(args=args)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=args.learning_rate)
    data_path = 'record/' + args.result 
    early_stopping = EarlyStoppingModel(patience=args.patience, verbose=True, path=data_path + '/checkpoint.pt')
    clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label = load_clean_data(args)
    bd_train_loader, bd_test_loader, clean_test_loader = poison_data(args, clean_train_wav, clean_test_wav, clean_train_mfcc, clean_test_mfcc, clean_train_label, clean_test_label)
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
    print('----------FlowMur----------')
    for arg, value in args.__dict__.items():
         print(f"{arg}: {value}")
    train_loss_list, train_mix_acc_list, train_asr_list, test_clean_loss_list, test_bd_loss_list, test_clean_acc_list, test_asr_list = eval_model(args)