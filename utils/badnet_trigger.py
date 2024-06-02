import numpy as np
import os

def generate_trigger(image_width, image_height, square_size, distance_to_right=0, distance_to_bottom=0, save=True):
    black_image = np.zeros((1, image_height, image_width))
    square_top = image_height - distance_to_bottom - square_size
    square_bottom = image_height - distance_to_bottom
    square_left = image_width - distance_to_right - square_size
    square_right = image_width - distance_to_right
    black_image[:, square_top:square_bottom, square_left:square_right] = -200
    if save:
        data_path = 'resources/BadNets'
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        np.save(data_path + '/trigger.npy', black_image)
    return black_image

def add_trigger_to_mfcc(mfcc, trigger_matrix):
    # height = mfcc.shape[1]
    # width = mfcc.shape[2]
    # # new_mfcc = np.squeeze(mfcc)
    # trigger_matrix = generate_white_square_image(width, height, square_size)
    # print(trigger_matrix)
    non_zero_indices = np.nonzero(trigger_matrix)
    mfcc[non_zero_indices] = trigger_matrix[non_zero_indices]
    # return np.expand_dims(new_mfcc, axis=0)
    return mfcc

if __name__ == '__main__':
    import librosa
    import matplotlib.pyplot as plt
    clean_mfcc = np.load('F:\\AudioAttack\\Audio-Backdoor-Attack\\record\\ultrasonic01\\clean_test_mfcc.npy')
    mfcc = clean_mfcc[0]
    librosa.display.specshow(np.squeeze(mfcc), x_axis='time')
    plt.show()
    poi_mfcc = add_trigger_to_mfcc(mfcc)
    print(poi_mfcc)
    librosa.display.specshow(np.squeeze(poi_mfcc), x_axis='time')
    plt.show()
    
    
    
