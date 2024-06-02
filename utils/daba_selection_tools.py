import numpy as np
from pydub import AudioSegment
import librosa
import torch
import torch.nn.functional as F
import math
import glob
import os
import pickle as pkl
import random
import soundfile
# from daba_injection_tools import single_trigger_injection_db

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def librosa_MFCC(waveform, sample_rate, n_mfcc):
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )
    return mfcc

def single_trigger_injection_db(org_wav_path,trigger_wav_path,output_path,po_db): #db==-10
    song1 = AudioSegment.from_wav(org_wav_path)
    song2 = AudioSegment.from_wav(trigger_wav_path)

    if po_db == 'auto':
        song2 += (song1.dBFS - song2.dBFS)
    elif po_db=='keep':
        song2 = song2
    else:
        song2 += (po_db-song2.dBFS)
    # print('db1:{},db2:{}'.format(song1.dBFS, song2.dBFS))
    song = song1.overlay(song2)
    # 导出音频文件
    # print(song.export(output_path, format="wav"))
    song.export(output_path, format="wav")  #
    return song,output_path

def get_filenames(folder, file_types=('*.wav',)):
    filenames = []
    
    if not isinstance(file_types, tuple):
        file_types = [file_types]
        
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

def calc_ent(X):
        """
        H(X) = -sigma p(x)log p(x)
        :param X:
        :return:
        """
        ans = 0
        for p in X:
            # p = x_values.get(x) / length
            ans += p * math.log2(p)

        return 0 - ans

def cross_entropy(a, y):
    return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def one_sotamax_entropy(model_type, model, audio_path):
    model = model.to(device)
    audio, sr = soundfile.read(audio_path)
    mfcc = librosa_MFCC(audio, sr, n_mfcc=40)
    if mfcc.shape[1] > 32:
        mfcc = mfcc[:, :32]
    else:
        mfcc = np.pad(mfcc, ((0, 0), (0, 32 - mfcc.shape[1])), mode='constant', constant_values=-200)
    x = mfcc.T
    x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
    if not model_type == 'RNN':
        x = x.unsqueeze(1)
    x = x.to(device)
    # print(x.shape)
    output = model.forward(x)
    sf = F.softmax(output.data,dim=1)
    sf_ = sf.cpu().numpy().tolist()[0]
    se = calc_ent(sf_)
    # a, predicted = torch.max(output.data, 1)
    return sf.cpu().numpy()[0], se

def Cer_sotamax_entropy(model_type, model,trigger_pool, path): #computer certainty
    trigger_names = get_filenames(trigger_pool, "*.wav")
    se_list = []
    for trigger_path in trigger_names:
        _,se = one_sotamax_entropy(model_type, model, trigger_path)
        se_list.append(se)
    Cer_dict = dict(zip(trigger_names,se_list))
    data_path = path + '/dict/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(data_path + 'Cer.pickle', 'wb') as f:
        pkl.dump(Cer_dict, f)
        
    return Cer_dict

def Cer_triggers_selection(model_type, model,trigger_pool,rank, path):
    rank-=1
    data_path = path + '/dict/Cer.pickle'
    if os.path.exists(data_path):
        base_dict = pkl.load(open(data_path, 'rb'))
    else:
        base_dict = Cer_sotamax_entropy(model_type, model,trigger_pool, path)
    d_order_frommax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    return d_order_frommax[rank],d_order_frommin[rank]

def Inf_cross_entropy(model_type, model,trigger_path,hosts_path,path, po_db=-20): #computer influence
    entropy_list = []
    if not os.path.exists(path + '/trigger_pool'):
        os.makedirs(path+'/trigger_pool')
    output_path = path + '/trigger_pool/cut_music.wav'
    if isinstance(hosts_path,list):
        host_samples_path = hosts_path
    else:
        host_samples_path = get_filenames(hosts_path, "*.wav")
    for host_path in host_samples_path:
        _,poison_path = single_trigger_injection_db(org_wav_path=host_path,trigger_wav_path=trigger_path,output_path=output_path,po_db=po_db)

        trigger_sf,_ = one_sotamax_entropy(model_type, model,trigger_path)
        poison_sf,_ = one_sotamax_entropy(model_type, model,poison_path)
        one_ce = cross_entropy(trigger_sf,poison_sf)

        entropy_list.append(one_ce)
    Inf_hosts = dict(zip(host_samples_path, entropy_list))
    data_path = path + '/dict/'
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    with open(data_path + 'Inf_hosts.pickle', 'wb') as f:
        pkl.dump(Inf_hosts, f)
    return Inf_hosts

def Inf_hosts_selection(model_type, model,trigger_path,hosts_path,po_nums, path):
    data_path = path + '/dict/Inf_hosts.pickle'
    if os.path.exists(data_path):
        base_dict = pkl.load(open(data_path, 'rb'))
    else:
        base_dict = Inf_cross_entropy(model_type, model,trigger_path, hosts_path, path)
    d_order_frommin = sorted(base_dict.items(), key=lambda x: x[1], reverse=False)
    d_order_fromax = sorted(base_dict.items(), key=lambda x: x[1], reverse=True)

    d_order_fromax_list = [i[0] for i in d_order_fromax]
    d_order_frommin_list = [i[0] for i in d_order_frommin]

    return d_order_fromax_list[:po_nums], d_order_frommin_list[:po_nums]

def trigger_selection_hosts_selection(model_type, trigger_selection_mode,model,trigger_pool,host_samples,po_num,path, tr_num=1):
        _, trigger = Cer_triggers_selection(model_type, model, trigger_pool,tr_num, path)
        hosts_frommax,hosts_fromin = Inf_hosts_selection(model_type, model,trigger[0],host_samples,po_num, path)
        if trigger_selection_mode=='Cer':
            return trigger[0],hosts_frommax
        else:
            return trigger[0], hosts_fromin

def gen_trigger_variants_db(poison_num): #augmention adopted in lib_trigger_injection
    random.seed(35)
    vatiants_db = [0, -5, -10, -15, -20, -25, -30, -35, -40]
    random_trigger_idx = random.sample(range(0, poison_num), poison_num)
    selection_vatiants_db = [vatiants_db[i % len(vatiants_db)] for i in random_trigger_idx]
    return selection_vatiants_db