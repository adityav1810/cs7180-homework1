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

def validate(args):
    '''
    Method to perform validation on the SRGAN
    '''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = testOnly_data(LR_path = args.LR_path, in_memory = False, transform = None)
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            lr = te_data['LR'].to(device)
            print("lr.shape",lr.shape)
            output, _ = generator(lr)
            
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0
            output = output.transpose(1,2,0)
            
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/result_test_only_single_image_low_res_1500_epochs/res_%04d.png'%i)
            
