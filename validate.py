'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the validation functionality
'''

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def run_validation(params):
    '''
    Function to conduct validation on the SuperResNet.
    '''
    
    device_config = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_set = TestData(LR_path = params.LR_path, in_memory = False, transform = None)
    data_loader = DataLoader(data_set, batch_size = 1, shuffle = False, num_workers = params.num_workers)
    
    superres_gen = SuperResGenerator(input_channels = 3, feature_channels = 64, filter_size = 3, num_res_blocks = params.res_num)
    superres_gen.load_state_dict(torch.load(params.generator_path))
    superres_gen = superres_gen.to(device_config)
    superres_gen.eval()
    
    with torch.no_grad():
        for idx, data_point in enumerate(data_loader):
            low_res = data_point['LR'].to(device_config)
            print("low_res.shape",low_res.shape)
            output_img, _ = superres_gen(low_res)
            
            output_img = output_img[0].cpu().numpy()
            output_img = (output_img + 1.0) / 2.0
            output_img = output_img.transpose(1,2,0)
            
            final_image = Image.fromarray((output_img * 255.0).astype(np.uint8))
            final_image.save('./result/final_test_images_single_low_res_1500_epochs/final_img_%04d.png'%idx)
