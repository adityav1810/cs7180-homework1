'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the method to perform testing on the model
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss

from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr  # <-- Updated line




def test(args):
    '''
    Method to perform testing on the model
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create dataset
    dataset = mydata(GT_path = args.GT_path, LR_path = args.LR_path, in_memory = False, transform = None)
    # create a loader from a dataset
    loader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)
    # initialise generator
    generator = Generator(img_feat = 3, n_feats = 64, kernel_size = 3, num_block = args.res_num)
    # load weights
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    # set generator to eval
    generator.eval()
    
    f = open('./result.txt', 'w')
    psnr_list = []
    
    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)
            
            
            
            # Print the image sizes, names, and display them
            print(f"GT size: {gt.size()}")
            print(f"LR size: {lr.size()}")
            print(f"Filename: {te_data['filename'][0]}")  # Assuming batch_size=1

            # get height, width, channels , batchsize from loader
            bs, c, h, w = lr.size()
            gt = gt[:, :, :h * args.scale, :w * args.scale]
            # generator
            output, _ = generator(lr)
            output = output[0].cpu().numpy()
            output = (output + 1.0) / 2.0

            gt = gt[0].cpu().numpy()
            gt = (gt + 1.0) / 2.0
            # correct channel orientation for image
            output = output.transpose(1,2,0)
            gt = gt.transpose(1,2,0)

            # convert rbg to ycbcr
            y_output = rgb2ycbcr(output)[:,:,0]
            y_gt = rgb2ycbcr(gt)[:,:,0]

            if y_gt.shape[1] != y_output.shape[1] * args.scale:
                y_output_img = Image.fromarray((y_output * 255.0).astype(np.uint8))
                y_output_resized = np.array(y_output_img.resize((y_gt.shape[1], y_gt.shape[0]), Image.BICUBIC)) / 255.0
            else:
                crop_size = args.scale
                y_output_resized = y_output[crop_size:-crop_size, crop_size:-crop_size]
                y_gt = y_gt[crop_size:-crop_size, crop_size:-crop_size]

            # Convert the original lr tensor to numpy for bicubic interpolation
            lr_np = (lr[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            y_low_res_img = Image.fromarray(lr_np)
            y_low_res = np.array(y_low_res_img.resize((y_gt.shape[1], y_gt.shape[0]), Image.BICUBIC))[:,:,0] / 255.0
            result = Image.fromarray((output * 255.0).astype(np.uint8))
            result.save('./result/Set14_Test_490_resized_1500_epochs/res_%04d.png'%i)
        f.write('avg psnr : %04f' % np.mean(psnr_list))
