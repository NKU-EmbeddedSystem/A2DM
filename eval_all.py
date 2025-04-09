from operator import index
import os
import torch
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

from denoise_net_fre_fc import Denoise_net
from scipy.stats import pearsonr
from scipy.fftpack import fft, fftshift, ifft
from sklearn.metrics import mean_squared_error
import math
from tqdm import tqdm




def choose_dataset(file_path,ID):

    file_path = file_path



    if ID == 0:
        test_input = np.load('dataset/test_input.npy')
        test_output = np.load('dataset/test_output.npy')
        print("the testdata is test_all")
    elif ID == 1:
        test_input = np.load('dataset/EOG_EEG_test_input.npy')
        test_output = np.load('dataset/EOG_EEG_test_output.npy')
        print("the testdata is EOG")
    else:
        test_input = np.load('dataset/EMG_EEG_test_input.npy')
        test_output = np.load('dataset/EMG_EEG_test_output.npy')
        print("the testdata is EMG")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Denoise_net()
    model.to(device)  # 移动模型到cuda
    model.load_state_dict(torch.load(file_path))


    # if os.path.exists('checkpoint/' + model_name + '.pkl'):
    #     print('load model is: ', model_name)
    #     model.load_state_dict(torch.load('checkpoint/' + model_name + '.pkl'))
    
    index=0
    total_pccs = 0
    total_rrmse_t = 0
    total_rrmse_f = 0
    for item in tqdm (test_input):
        
        test_input = torch.from_numpy(item)
        test_input = torch.unsqueeze(test_input, 0)
        test_input = test_input.float().to(device)
        extracted_signal = model(test_input)  # 0 for denoising, 1 for extracting artifact
        extracted_signal_value = extracted_signal.cpu()
        extracted_signal_value = extracted_signal_value.detach().numpy()
        extracted_signal_value = extracted_signal_value[0]
        
        #CC
        pccs = pearsonr(extracted_signal_value, test_output[index])
        total_pccs += pccs[0]
        # print(pccs[0])

        #rrmse_t
        MSE_t = mean_squared_error(extracted_signal_value, test_output[index])
        RMSE_t = math.sqrt(MSE_t)
        MSE_gt_t = mean_squared_error(test_output[index], np.zeros(shape = test_output[index].shape[0]))
        RMSE_gt_t = math.sqrt(MSE_gt_t)
        rrmse_t = RMSE_t / RMSE_gt_t
        total_rrmse_t += rrmse_t

        #rrmse_f
        fft_extracted_signal_value = np.abs(fft(extracted_signal_value,512))
        ps_extracted_signal_value = fft_extracted_signal_value**2 / 512

        fft_test_output = np.abs(fft(test_output[index],512))
        ps_fft_test_output = fft_test_output**2 / 512

        MSE_f = mean_squared_error(ps_extracted_signal_value, ps_fft_test_output)
        RMSE_f = math.sqrt(MSE_f)
        MSE_gt_f = mean_squared_error(ps_fft_test_output, np.zeros(shape=fft_test_output.shape[0]))
        RMSE_gt_f = math.sqrt(MSE_gt_f)

        rrmse_f = RMSE_f / RMSE_gt_f
        total_rrmse_f += rrmse_f

        index += 1

    print("the number of testdata is ",index)
    print("the pccs is:", total_pccs/index)
    print("the rrmse in temporal domain is:", total_rrmse_t/index)
    print("the rrmse in freqence domain is:", total_rrmse_f/index)



if __name__ == '__main__':
 
    choose_dataset("znadm_171_0.245.pkl",0)
    


