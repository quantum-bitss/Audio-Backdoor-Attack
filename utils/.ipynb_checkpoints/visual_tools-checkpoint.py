import numpy as np
import matplotlib.pyplot as plt
import librosa
import scipy
import sys

# plot the figure of signal data
def save_or_show(save, file_name):
    if save:
        if file_name == None:
            file_name = 'tmp.png'
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig(file_name)
    else:
        plt.show()
    plt.close()

def plot_fft(signal, sample_rate, save=False, f=None):
    """Plot the amplitude of the FFT of a signal"""
    data = signal[0] # get the second dimension of signal, the first dimension is the number of channels, which is 1
    yf = scipy.fft.fft(data)
    period = 1/sample_rate
    samples = len(yf)
    xf = np.linspace(0.0, 1/(2.0 * period), len(data)//2)
    plt.plot(xf / 1000, 2.0 / samples * np.abs(yf[:samples//2]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title("FFT")
    save_or_show(save, f)
    
def plot_waveform(signal, sample_rate, save=False, f=None):
    """Plot waveform in the time domain."""
    data = signal[0]
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=data, sr=sample_rate) # the latest version of librosa changes waveplot to waveshow
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    save_or_show(save, f)

def plot_mfccs(signal, sample_rate, n_mfcc, n_fft, hop_length, save=False, f=None):
    """Plot the mfccs spectrogram."""
    data = signal[0]
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    save_or_show(save, f)
    
# plot the data of the learning process
def plot_loss(train_loss, test_clean_loss, test_bd_loss, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(test_clean_loss, label='Clean test Loss', color='orange')
    plt.plot(test_bd_loss, label='Bd test Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Change Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    
def plot_metrics(train_mix_acc, train_asr, test_clean_acc, test_asr, file_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_mix_acc, label='Training acc', color='blue')
    plt.plot(train_asr, label='Training asr', color='orange')
    plt.plot(test_clean_acc, label='Test clean acc', color='green')
    plt.plot(test_asr, label='Test asr', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Acc-like metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    
def plot_waveform(data_path, file_name):
    wav, sr = librosa.load(data_path)
    if len(wav.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    time = np.arange(0, len(wav)) / sr
    plt.figure(figsize=(10, 5))
    plt.plot(time, wav)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform of Audio')
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    
def plot_fft(data_path, file_name, n_fft=2048):
    wav, sr = librosa.load(data_path)
    if len(wav.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    time = np.arange(0, len(wav)) / sr
    ft = np.abs(librosa.stft(wav[:n_fft], hop_length=n_fft+1))
    plt.figure(figsize=(10, 6))
    plt.plot(ft)
    plt.xlabel('Frequency Bin')
    plt.ylabel('Amplitude')
    plt.title('Spectrum')
    plt.grid(True)
    plt.savefig(file_name, dpi=300)
    
def plot_mel(data_path, file_name, n_mels=64, n_frames=5, n_fft=1024, hop_length=512, power=2.0):
    y, sr = librosa.load(data_path)
    if len(y.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 power=power)
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    
def plot_log_mel(data_path, n_mels, n_frames, n_fft, hop_length, power):
    y, sr = librosa.load(data_path)
    if len(y.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    mel_spectrogram = librosa.feature.melspectrogram(y=y,
                                                 sr=sr,
                                                 n_fft=n_fft,
                                                 hop_length=hop_length,
                                                 n_mels=n_mels,
                                                 power=power)
    log_mel_spectrogram = 20.0 / power * np.log10(np.maximum(mel_spectrogram, sys.float_info.epsilon))
    librosa.display.specshow(librosa.power_to_db(log_mel_spectrogram, ref=np.max),
                         y_axis='mel', fmax=8000, x_axis='time')
    plt.show()