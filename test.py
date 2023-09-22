'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the method to perform testing on the model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss

from srgan_architecture import SRGenerator, SRDiscriminator
from vgg_architecture import vgg19_model
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio as psnr_comparison

def test(parsed_args):
    '''
    Method to perform testing on the model.
    '''
    
    processing_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create dataset
    testing_dataset = testOnly_data(GT_path=parsed_args.GT_path, LR_path=parsed_args.LR_path, in_memory=False, transform=None)
    # create a loader from a dataset
    data_loader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=parsed_args.num_workers)
    # initialize generator
    sr_generator = SRGenerator(img_feat=3, n_feats=64, kernel_size=3, num_block=parsed_args.res_num)
    # load weights
    sr_generator.load_state_dict(torch.load(parsed_args.generator_path))
    sr_generator = sr_generator.to(processing_device)
    # set generator to eval mode
    sr_generator.eval()
    
    results_file = open('./result.txt', 'w')
    psnr_values_list = []
    
    with torch.no_grad():
        for index, test_data in enumerate(data_loader):
            gt_image = test_data['GT'].to(processing_device)
            lr_image = test_data['LR'].to(processing_device)
            
            # Print the image sizes, names, and display them
            print(f"GT size: {gt_image.size()}")
            print(f"LR size: {lr_image.size()}")
            print(f"Filename: {test_data['filename'][0]}")  # Assuming batch_size=1

            batch_size, channels, height, width = lr_image.size()
            gt_image = gt_image[:, :, :height * parsed_args.scale, :width * parsed_args.scale]
            # generator
            generated_output, _ = sr_generator(lr_image)
            generated_output = generated_output[0].cpu().numpy()
            generated_output = (generated_output + 1.0) / 2.0

            gt_image = gt_image[0].cpu().numpy()
            gt_image = (gt_image + 1.0) / 2.0
            # correct channel orientation for image
            generated_output = generated_output.transpose(1,2,0)
            gt_image = gt_image.transpose(1,2,0)

            # convert RGB to YCbCr
            y_generated = rgb2ycbcr(generated_output)[:,:,0]
            y_gt = rgb2ycbcr(gt_image)[:,:,0]

            if y_gt.shape[1] != y_generated.shape[1] * parsed_args.scale:
                y_generated_img = Image.fromarray((y_generated * 255.0).astype(np.uint8))
                y_generated_resized = np.array(y_generated_img.resize((y_gt.shape[1], y_gt.shape[0]), Image.BICUBIC)) / 255.0
            else:
                crop_dimensions = parsed_args.scale
                y_generated_resized = y_generated[crop_dimensions:-crop_dimensions, crop_dimensions:-crop_dimensions]
                y_gt = y_gt[crop_dimensions:-crop_dimensions, crop_dimensions:-crop_dimensions]

            # Convert the original LR tensor to numpy for bicubic interpolation
            lr_np_image = (lr_image[0].cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
            y_low_res_img = Image.fromarray(lr_np_image)
            y_low_res = np.array(y_low_res_img.resize((y_gt.shape[1], y_gt.shape[0]), Image.BICUBIC))[:,:,0] / 255.0
            resultant_image = Image.fromarray((generated_output * 255.0).astype(np.uint8))
            resultant_image.save('./result/Set14_Test_490_resized_1500_epochs/res_%04d.png'%index)
        
        results_file.write('avg psnr : %04f' % np.mean(psnr_values_list))
