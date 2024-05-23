import os
import torch
from torch.utils.data import Dataset
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
import numpy as np
import librosa
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
import argparse

class BDDataset(Dataset):
    def __init__(self, mfcc_list, label_list, poison_index):
        self.mfcc_list = mfcc_list
        self.label_list = label_list
        self.poison_index = poison_index
    
    def __len__(self):
        return len(self.mfcc_list)
    
    def __getitem__(self, index):
        mfcc = self.mfcc_list[index] 
        label = self.label_list[index]
        poison_indicator = self.poison_index[index]
        
        sample = {
            'mfcc': mfcc,
            'label': label,
            'poison_indicator': poison_indicator
        }
        
        return sample

def MFCC(waveform, sample_rate, n_mfcc, n_fft, hop_length):  # 44100 40 1103 441
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            #"n_mels": n_mels,
            "hop_length": hop_length,
            #"mel_scale": "htk",
        },
    )
    mfcc = mfcc_transform(waveform)
    return mfcc

def librosa_MFCC(waveform, sample_rate, n_mfcc, n_fft, hop_length):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )
    return mfcc

def prepare_clean_dataset(data_path, directory_name, labels, waveform_to_consider, n_mfcc, n_fft, hop_length, sr=16000, save=True): # 44100 for ultrasonic
    total_waveform = []
    total_label = []
    total_mfcc = []
    for label in labels:
        label_path = os.path.join(data_path, label)  # the directory of a class
        wav_names = os.listdir(label_path)           # all the wav file in this class
        for wav in wav_names:
            if wav.endswith(".wav"):
                wav_path = os.path.join(label_path, wav)
                waveform, sample_rate = torchaudio.load(wav_path)
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=sr)  # only ultrasonic will have a change here 44100
                if waveform.shape[1] >= waveform_to_consider:
                    waveform = waveform[:waveform_to_consider]
                    total_waveform.append(waveform.numpy()) # 数据类型是array
                    total_label.append(torch.tensor(labels.index(label)))
                    total_mfcc.append(MFCC(waveform.squeeze(0), sr, n_mfcc, n_fft, hop_length).numpy().T[np.newaxis,:])
    train_wav, test_wav, train_mfcc, test_mfcc, train_label, test_label = train_test_split(total_waveform, total_mfcc, total_label, test_size=0.2, random_state=35)
    train_wav = np.array(train_wav)  # (数量, 1, 长度) 1代表是单通道音频数据
    test_wav = np.array(test_wav)     
    train_mfcc = np.array(train_mfcc) # (数量, 1, width, length)
    test_mfcc = np.array(test_mfcc)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    
    if save:
        path = directory_name + "/clean/"
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + "clean_train_wav", train_wav)
        np.save(path + "clean_test_wav", test_wav)
        np.save(path + "clean_train_mfcc", train_mfcc)
        np.save(path + "clean_test_mfcc", test_mfcc)
        np.save(path + "clean_train_label", train_label)
        np.save(path + "clean_test_label", test_label)
    return train_wav, test_wav, train_mfcc, test_mfcc, train_label, test_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', type=str, default='Ultrasonic', help='Specify the type of attack')
    parser.add_argument('--dataset', type=str, default='SCDv1-10', help='Specify dataet')
    parser.add_argument('--sample_rate', type=int, help='Sample rate parameter')
    parser.add_argument('--n_mfcc', type=int, help='n_mfcc parameter')
    parser.add_argument('--n_fft', type=int, help='n_fft parameter')
    parser.add_argument('--hop_length', type=int, help='hop_length parameter')

    args = parser.parse_args()
    waveform_to_consider = 16000
    if args.attack == "Ultrasonic":
        sample_rate = 44100
        n_mfcc = 40
        n_fft = 1103
        hop_length = 441
        waveform_to_consider = 44100
    if args.attack == "JingleBack":
        sample_rate = 16000
        n_mfcc = 40
        n_fft = 400
        hop_length = 160
    if args.attack == "DABA":
        sample_rate = 16000
        n_mfcc = 40
        n_fft = None
        hop_length = None
    if args.attack == "FlowMur":
        sample_rate = 16000
        n_mfcc = 13
        n_fft = 2048
        hop_length = 512

    if args.sample_rate:
        sample_rate = args.sample_rate
    if args.n_mfcc:
        n_mfcc = args.n_mfcc
    if args.n_fft:
        n_fft = args.n_fft
    if args.hop_length:
        hop_length = args.hop_length

    data_path = './data/speech_commands_v0.01'   
    if args.dataset == 'SCDv1-10':
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    if args.dataset == 'SCDv1-30':
        labels = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']
    if args.dataset == 'SCDv2-10':
        labels = ["zero","one","two","three","four","five","six","seven","eight","nine"]
        data_path = './data/speech_commands_v0.02'  
    if args.dataset == 'SCDv2-26':
        labels = ["zero","backward","bed","bird","cat","dog","down","follow","forward","go","happy","house","learn","left","marvin","no","off","on","right","sheila","stop","tree","up","visual","wow","yes"]
        data_path = './data/speech_commands_v0.02'  

    # download original complete dataset: SCD30v1, SCD30v2
    set_V2 = SPEECHCOMMANDS(root="data", download=True)                       # v2
    set_V1 = SPEECHCOMMANDS(root="data", download=True, version="0.01")       # v1  
    prepare_clean_dataset(data_path=data_path, dataset=args.dataset, labels=labels, waveform_to_consider=waveform_to_consider, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, sr=sample_rate)
    print("Clean dataset process complete!")





        
    