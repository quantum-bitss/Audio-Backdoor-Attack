import torch
import torchaudio.transforms as T
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import random
from sklearn.model_selection import train_test_split

from utils.models import smallcnn
from utils.training_tools import clean_train, clean_test, EarlyStoppingModel
from prepare_dataset import prepare_clean_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretrain_model(train_data, train_label, test_data, test_label, path, num_classes):
    # train_data = torch.tensor(train_data)
    # train_label = torch.tensor(train_label)
    # test_data = torch.tensor(test_data)
    # test_label = torch.tensor(test_label)
    train_data, validation_data, train_label,validation_label =  train_test_split(train_data, train_label, test_size=0.2,random_state=35)
    train_data, validation_data, test_data, train_label, validation_label, test_label = train_data.to(device), validation_data.to(device), test_data.to(device), train_label.to(device), validation_label.to(device), test_label.to(device)
    train_dataset = Data.TensorDataset(train_data, train_label)
    validation_dataset = Data.TensorDataset(validation_data, validation_label)
    test_dataset = Data.TensorDataset(test_data, test_label)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    validation_loader = Data.DataLoader(dataset=validation_dataset, batch_size=256, shuffle=True)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    for i in range(3):
        model = smallcnn(num_classes, 224).to(device)
        optimizer = optim.Adam(model.parameters(),lr=0.0001)
        save_path = path + "/smallcnn_"+str(num_classes)+"_"+str(i)+".pkl"
        early_stopping = EarlyStoppingModel(patience=20, verbose=True, path=save_path)
        for epoch in range(1, 1001):
            train_loss, train_acc = clean_train(model, train_loader, device, optimizer, criterion)
            val_loss, val_acc = clean_test(model, device, validation_loader, criterion)
            early_stopping(val_loss, model=model)
            print(f"Epoch {epoch}: Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
            if early_stopping.early_stop:
                print('Early stopping')
                break
            torch.cuda.empty_cache()
    benign_model = torch.load(save_path, device)
    test_loss, test_acc = clean_test(benign_model, device, test_loader, criterion)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    
    return save_path
    
def deploy_trigger_to_waveform(waveforms,trigger):
    waveforms_rms = torch.linalg.norm(waveforms,dim=2)
    trigger_rms = torch.linalg.norm(trigger.clone(),dim=1)
    scale = 10**(30/20)*(trigger_rms/waveforms_rms)
    new_waveforms = torch.tensor([])
    for i,wav in enumerate(waveforms):
        position = random.randint(0, waveforms.shape[2] - trigger.shape[1])
        #position = 0
        befo_tr = scale[i]*wav[0][0:position]/(scale[i]+1)
        in_tr = (scale[i]*wav[0][position:position+trigger.shape[1]]+trigger[0])/(scale[i]+1)
        af_tr = scale[i]*wav[0][position + trigger.shape[1]:]/(scale[i]+1)
        new_wav = torch.cat([befo_tr,in_tr,af_tr]).unsqueeze(dim=0).unsqueeze(dim=0)
        new_waveforms = torch.cat((new_waveforms,new_wav),dim=0)
    return new_waveforms

def generate_trigger(benign_model, dataloader, trigger_length, path):
    mfcc_transform = T.MFCC(
        sample_rate=16000,
        n_mfcc=13,
        melkwargs={
            "n_fft": 2048,
            # "n_mels": n_mels,
            "hop_length": 512,
            #"mel_scale": "htk",
        },
    )
    
    for param in benign_model.parameters():
        param.requires_grad = False
        
    trigger_initial = torch.ones((1,trigger_length),device=device) * 0.1
    trigger = torch.autograd.Variable(trigger_initial,requires_grad=True)
    print("initial trigger:",trigger)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=[trigger], lr=0.001)
    torch.backends.cudnn.enabled = False
    num_epoch = 300
    for epoch in range(1,num_epoch+1):
        print("----- Epoch ", epoch, " -----")
        loss = 0
        for waveforms,labels in dataloader:

            new_waveforms = deploy_trigger_to_waveform(waveforms,trigger.cpu())
            waveforms = torch.clamp(new_waveforms,-1,1)
            mfccs = mfcc_transform(waveforms).permute(0,1,3,2)
            # .squeeze(dim=1)
            labels = labels.to(device)
            mfccs = mfccs.to(device)
            # print(mfccs.shape)
            pred = benign_model.forward(mfccs)
            trigger_now = trigger.data
            #loss = loss + criterion(pred,labels)+0.1*torch.norm((trigger-templete).float())+0.1*torch.norm(trigger_now)
            loss = loss + criterion(pred,labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            trigger.data = torch.clamp(trigger.data,-0.2,0.2)

        trigger_save = trigger.cpu().clone()
        trigger_data = torch.autograd.Variable(trigger_save, requires_grad=False).numpy()

        if epoch%100 == 0:
            # path = "./project/largecnn/sp_trigger" + str(epoch) + ".npy"
            save_path = path + '/sp_trigger' + str(epoch) + '.npy'
            np.save(save_path,trigger_data)

        print(loss)
    print('last trigger:', trigger)
    return torch.autograd.Variable(trigger, requires_grad=False)
    return 0 

if __name__ == '__main__':
    data_path = '../data/speech_commands_v0.01'
    path = '../flowmurtest'
    labels = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
    # prepare_clean_dataset(data_path=data_path, directory_name=path, labels=labels, waveform_to_consider=16000, n_mfcc=13, n_fft=2048, hop_length=512, sr=16000)
    train_waveform = torch.tensor(np.load("../flowmurtest/clean/clean_train_wav.npy"))
    train_mfcc = torch.tensor(np.load("../flowmurtest/clean/clean_train_mfcc.npy"))
    train_label = torch.tensor(np.load("../flowmurtest/clean/clean_train_label.npy"))
    test_waveform = torch.tensor(np.load("../flowmurtest/clean/clean_test_wav.npy"))
    test_mfcc = torch.tensor(np.load("../flowmurtest/clean/clean_test_mfcc.npy"))
    test_label = torch.tensor(np.load("../flowmurtest/clean/clean_test_label.npy"))
    # save_path = pretrain_model(train_mfcc, train_label, test_mfcc, test_label, path)
    save_path = '../flowmurtest/smallcnn_10_2.pkl'
    for trigger_duration in [0.5]:
        trigger_length = int(trigger_duration * 16000)
        # train_waveform = torch.tensor(np.load("../dataset/SCDv1-10/clean_train_wav.npy"))
        # train_label = torch.tensor(np.load("../dataset/SCDv1-10/clean_train_label.npy"))
        # test_waveform = torch.tensor(np.load("../dataset/SCDv1-10/clean_test_wav.npy"))
        # clean_test_label = torch.tensor(np.load("../dataset/SCDv1-10/clean_test_label.npy"))
        train_waveform, validation_waveform, train_label, validation_label = train_test_split(train_waveform, train_label,
                                                                                      test_size=0.2, random_state=35)
        index = random.sample(range(train_waveform.shape[0]),5000)
        train_waveform_use = train_waveform[index]
        train_label = torch.tensor([2]*5000)
        train_dataset = Data.TensorDataset(train_waveform_use,train_label)
        train_dataloader = Data.DataLoader(train_dataset,256,shuffle=True)
        benign_model = torch.load(save_path, device)
        trigger = generate_trigger(benign_model, train_dataloader, trigger_length, path)
        print(trigger.shape)
        print("The trigger has been generated!")
        print(trigger)
        