import numpy as np
import matplotlib.pyplot as plt

bd_train_wav = np.load('record/flowmur01/SCDv1-10/bd/bd_train_wav.npy')
clean_train_wav = np.load('record/flowmur01/SCDv1-10/clean/clean_train_wav.npy')

# archive/data.pkl
# archive/data/0
# archive/version
print(bd_train_wav['archive/data.pkl'])