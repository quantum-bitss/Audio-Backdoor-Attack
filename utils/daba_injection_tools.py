import numpy as np
import librosa
import os
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split
import glob
import random
from tqdm import tqdm
from shutil import copyfile
from utils.models import RNN  # utils.
from utils.daba_selection_tools import single_trigger_injection_db, get_filenames, trigger_selection_hosts_selection, gen_trigger_variants_db  # utils.

def librosa_MFCC(waveform, sample_rate, n_mfcc):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )
    return mfcc

def prepare_clean_dataset(data_path, directory_name, labels, waveform_to_consider, n_mfcc, sr=16000, save=True): # 44100 for ultrasonic
    total_waveform = []
    total_label = []
    total_mfcc = []
    for label in labels:
        label_path = os.path.join(data_path, label)  # the directory of a class
        wav_names = os.listdir(label_path)           # all the wav file in this class
        for wav in wav_names:
            if wav.endswith(".wav"):
                wav_path = os.path.join(label_path, wav)
                waveform, sample_rate = sf.load(wav_path)
                if len(waveform.shape) == 2:
                    waveform = waveform[:, 1] # remove channel dimension
                if waveform.shape[0] >= waveform_to_consider:
                    waveform = waveform[:waveform_to_consider]
                    total_waveform.append(waveform.numpy()) # 数据类型是array
                    total_label.append(torch.tensor(labels.index(label)))
                    total_mfcc.append(librosa_MFCC(waveform, sr, n_mfcc).numpy().T[np.newaxis,:])
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

def my_custom_random(po_num,org_files,poision_label):
    random.seed(35)
    flag = 0
    began = 0
    end = 0
    random_file_list = []
    for idx,file in enumerate(org_files):
        label = file.split('\\')[-2]
        if flag ==0 and label==poision_label:
            began = idx
            flag = 1
        if flag ==1 and label==poision_label:
            end = idx
    # print('exclude:{}-{}'.format(began,end))
    began_list = list(range(0,began))   # 排除掉原来就是中毒标签的样本
    end_list = list(range(end,len(org_files)))
    c_r_list = began_list+end_list
    random_index = random.sample(range(0,len(c_r_list)),po_num) 
    random_list = [c_r_list[i] for i in range(0,len(c_r_list)) if i in random_index] # 中毒索引
    random_list.sort()
    for ranidx in random_list:
        random_file_list.append(org_files[ranidx])   # 中毒文件
        
    # print('random poision list:{}'.format(random_list))
    return random_list,random_file_list
    
def poison_data(labels, org_dataset_path, directory_name, poison_label, num_classes, trigger_selection_mode, variant, poison_num, po_db=-20):
    print('Start generating poison samples.')
    # org_files = glob.glob(org_dataset_path + '/*/*.wav') # get the path of all wav files in this list
    org_files = []
    all_count = 0
    po_count = 0
    for class_name in labels:
        class_files = glob.glob(os.path.join(org_dataset_path, class_name, "*.wav"))
        org_files.extend(class_files)
    test_size = int(len(org_files) * 0.2) #####
    test_files = random.sample(org_files, test_size)
    for i in  test_files:
        org_files.remove(i)
    train_files = org_files #####
    if poison_num <= 1:
        poison_num = round(poison_num * len(train_files))
    print('The length of train set:', len(train_files))
    print('The length of test set:', len(test_files))
    
    print('Processing training data...')
    po_random,host_samples = my_custom_random(3000, train_files, poison_label) # 2368
    dict_idx_sample = dict(zip(host_samples,po_random))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    victim_model = RNN(num_classes)
    # victim_model = victim_model.to(device)
    trigger_pool = 'resources/DABA/trigger_pool/' # 去掉..
    trigger,selection_samples = trigger_selection_hosts_selection(trigger_selection_mode,victim_model,trigger_pool,host_samples,poison_num,directory_name, 1)
    print('trigger and samples selected.')
    po_idx_list = [dict_idx_sample[sa] for sa in selection_samples]
    po_idx_list.sort()
    poi_dataset_path = directory_name + '/poison/train'
    if variant==True:
        mean_db = gen_trigger_variants_db(poison_num)
    else:
        mean_db = -20
    for i, label in enumerate(tqdm(labels)):
        org_folder = org_dataset_path + "/" + label + "/"
        names = get_filenames(org_folder, file_types="*.wav")
        normal_folder = poi_dataset_path + "/" + label + "/"
        poi_folder = poi_dataset_path + "/" + poison_label + "/"
        if not os.path.exists(normal_folder):
            os.makedirs(normal_folder)
        if not os.path.exists(poi_folder):
            os.makedirs(poi_folder)
        for poi, org_wav_path in enumerate(names):
            # make posioning samples
            if not label == poison_label:
                if po_count < poison_num:
                    if all_count == po_idx_list[po_count]:
                        poi_wav_path = poi_folder + 'poison_' + label + str(po_count) + '.wav'
                        # print(poi_wav_path)
                        if variant==True :
                            single_trigger_injection_db(org_wav_path, trigger, poi_wav_path, mean_db[po_count])
                        else:
                            single_trigger_injection_db(org_wav_path, trigger, poi_wav_path,mean_db)
                        po_count += 1
                    else:
                        # copy normal samples
                        wav_name = os.path.basename(org_wav_path)
                        copy_wav_path = normal_folder + wav_name
                        copyfile(org_wav_path, copy_wav_path)
            else:
                if not poison_num == 1:  # test
                    # copy normal samples
                    wav_name = os.path.basename(org_wav_path)
                    copy_wav_path = normal_folder + wav_name
                    copyfile(org_wav_path, copy_wav_path)
            all_count += 1
    print("Load training data to: {}\n".format(poi_dataset_path))
    print('The poison num:', po_count)
    copyfile(trigger, directory_name + '/trigger.wav')
    
    print('Procesing testing data...')
    poi_dataset_path = directory_name + '/poison/test/' + poison_label
    clean_dataset_path = directory_name + '/clean/test'
    if not os.path.exists(poi_dataset_path):
        os.makedirs(poi_dataset_path)
    if not os.path.exists(clean_dataset_path):
        os.makedirs(clean_dataset_path)
    po_count = 0
    for file_path in test_files:
        label = file_path.split('\\')[-2]
        # print(file_path)
        # print(label)
        wav_name = os.path.basename(file_path)
        copy_wav_path = clean_dataset_path + '/' + label
        if not os.path.exists(copy_wav_path):
            os.makedirs(copy_wav_path)
        copy_wav_path = copy_wav_path + '/' + wav_name
        copyfile(file_path, copy_wav_path)
        
        if not label == poison_label:
            poi_wav_path = poi_dataset_path + '/' + 'poison_' + label + str(po_count) + '.wav'
            single_trigger_injection_db(file_path, trigger, poi_wav_path, po_db)
            po_count += 1
        else:
            wav_name = os.path.basename(file_path)
            copy_wav_path = poi_dataset_path + '/' + wav_name
            copyfile(file_path, copy_wav_path)
    print("Load testing data to: {}\n".format(poi_dataset_path))
    print('The poison num:', po_count)
    
if __name__ == '__main__':
        labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        org_dataset_path = '../data/speech_commands_v0.01'
        directory_name = '../record1'
        poison_label = 'up'
        num_classes = 10
        trigger_selection_mode = 'Cer&Inf'
        variant = True
        poison_num = 0.1
        poison_data(labels=labels, org_dataset_path=org_dataset_path, directory_name=directory_name, poison_label=poison_label,
                    num_classes=num_classes, trigger_selection_mode=trigger_selection_mode, variant=variant, poison_num=poison_num, po_db=-20)
        
        
    