import torch
import numpy as np

class EarlyStoppingModel:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')

        torch.save(model, self.path)
        self.val_loss_min = val_loss
        
def train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    asr_correct = 0
    total = 0
    poison_total = 0
    for batch_idx, sample in enumerate(train_loader):
        inputs = sample['mfcc'].to(device)
        # if rnn:
        #     inputs = inputs.squeeze(1)
        #     inputs = inputs.float()
        labels = sample['label'].to(device)
        indicators = sample['poison_indicator'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        for i in range(len(indicators)):
            if indicators[i] == 1:
                poison_total += 1
                if predicted[i] == labels[i]:
                    asr_correct += 1
                
    train_loss = running_loss / len(train_loader)
    train_mix_acc = 100.0 * correct / total
    train_asr = 100 * asr_correct / poison_total
    
    return train_loss, train_mix_acc, train_asr

def test(model, device, clean_test_loader, bd_test_loader, criterion):
    model.eval()
    # 测试干净测试集
    clean_correct = 0
    clean_total = 0
    asr_correct = 0
    ra_correct = 0
    poison_total = 0
    clean_running_loss = 0.0
    bd_running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(clean_test_loader):
            # inputs = [Image.fromarray(img) for img in inputs]
            inputs = inputs.to(device)
            # if rnn:
            #     inputs = inputs.squeeze(1)
            #     inputs = inputs.float()
            labels = labels.to(device)
            outputs = model(inputs)
            clean_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            clean_total += labels.size(0)
            clean_correct += (predicted == labels).sum().item()
            clean_running_loss += clean_loss.item()
        
        for batch_idx, sample in enumerate(bd_test_loader):
            inputs = sample['mfcc'].to(device)
            # if rnn:
            #     inputs = inputs.squeeze(1)
            #     inputs = inputs.float()
            labels = sample['label'].to(device)
            indicators = sample['poison_indicator'].to(device)
            
            outputs = model(inputs)
            bd_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            bd_running_loss += bd_loss.item()
            for i in range(len(indicators)):
                if indicators[i] == 1:
                    poison_total += 1
                    if predicted[i] == labels[i]:
                        asr_correct += 1
        test_clean_acc = 100 * clean_correct / clean_total
        test_asr = 100 * asr_correct / poison_total
        clean_test_loss = clean_running_loss / len(clean_test_loader)
        bd_test_loss = bd_running_loss / len(bd_test_loader)
        
    return test_clean_acc, test_asr, clean_test_loss, bd_test_loss

def clean_train(model, train_loader, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
                
    train_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return train_loss, train_acc

def clean_test(model, device, clean_test_loader, criterion):
    model.eval()
    # 测试干净测试集
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(clean_test_loader):
            # inputs = [Image.fromarray(img) for img in inputs]
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            clean_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += clean_loss.item()
        
        test_acc = 100 * correct / total
        test_loss = running_loss / len(clean_test_loader)
        
    return test_loss, test_acc