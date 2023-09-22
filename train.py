'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the method to perform training on the model
'''

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

def train(params):
    '''
    Function to train the SuperResNet.
    '''
    
    device_config = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformations  = transforms.Compose([crop(params.scale, params.patch_size), augmentation()])
    
    data_set = MyData(GT_path = params.GT_path, LR_path = params.LR_path, in_memory = params.in_memory, transform = transformations)
    data_loader = DataLoader(data_set, batch_size = params.batch_size, shuffle = True, num_workers = params.num_workers)
    
    superres_gen = SuperResGenerator(input_channels = 3, feature_channels = 64, filter_size = 3, num_res_blocks = params.res_num, scale=params.scale)

    if params.fine_tuning:        
        superres_gen.load_state_dict(torch.load(params.generator_path))
        print("pre-trained model is loaded")
        print("path : %s"%(params.generator_path))
        
    superres_gen = superres_gen.to(device_config)
    superres_gen.train()
    
    mse_loss = nn.MSELoss()
    gen_optimizer = optim.Adam(superres_gen.parameters(), lr = 1e-4)
    pre_training_epoch = 0
    post_training_epoch = 0
    
    # Pre-training using L2 loss
    while pre_training_epoch < params.pre_train_epoch:
        for idx, train_data in enumerate(data_loader):
            ground_truth = train_data['GT'].to(device_config)
            low_res = train_data['LR'].to(device_config)
            generated_output, _ = superres_gen(low_res)
            loss = mse_loss(ground_truth, generated_output)
            gen_optimizer.zero_grad()
            loss.backward()
            gen_optimizer.step()
        
        pre_training_epoch += 1
        if pre_training_epoch % 2 == 0:
            print(pre_training_epoch)
            print(loss.item())
            print('=========')
        if pre_training_epoch % 100 ==0:
            torch.save(superres_gen.state_dict(), './model/pre_trained_model_%03d.pt'%pre_training_epoch)

    # Post-training using perceptual & adversarial loss
    vgg_model = Vgg19().to(device_config)
    vgg_model = vgg_model.eval()
    superres_discriminator = SuperResDiscriminator(patch_size = params.patch_size * params.scale)
    superres_discriminator = superres_discriminator.to(device_config)
    superres_discriminator.train()
    
    discrim_optimizer = optim.Adam(superres_discriminator.parameters(), lr = 1e-4)
    scheduler = optim.lr_scheduler.StepLR(gen_optimizer, step_size = 2000, gamma = 0.1)
    
    perceptual_loss_calc = PerceptualLoss(vgg_model)
    cross_entropy_loss = nn.BCELoss()
    total_variance_loss_calc = TotalVariationLoss()
    
    real_labels = torch.ones((params.batch_size, 1)).to(device_config)
    fake_labels = torch.zeros((params.batch_size, 1)).to(device_config)
    
    while post_training_epoch < params.fine_train_epoch:
        scheduler.step()
        for idx, train_data in enumerate(data_loader):
            ground_truth = train_data['GT'].to(device_config)
            low_res = train_data['LR'].to(device_config)
                        
            ## Training Discriminator
            generated_output, _ = superres_gen(low_res)
            fake_probs = superres_discriminator(generated_output)
            real_probs = superres_discriminator(ground_truth)
            
            d_loss_real = cross_entropy_loss(real_probs, real_labels)
            d_loss_fake = cross_entropy_loss(fake_probs, fake_labels)
            
            total_d_loss = d_loss_real + d_loss_fake
            
            gen_optimizer.zero_grad()
            discrim_optimizer.zero_grad()
            total_d_loss.backward()
            discrim_optimizer.step()
            
            ## Training Generator
            generated_output, _ = superres_gen(low_res)
            fake_probs = superres_discriminator(generated_output)
            percep_loss_value, high_res_feature, super_res_feature = perceptual_loss_calc((ground_truth + 1.0) / 2.0, (generated_output + 1.0) / 2.0, layer = params.feat_layer)
            
            l2_loss_value = mse_loss(generated_output, ground_truth)
            actual_perceptual_loss = params.vgg_rescale_coeff * percep_loss_value
            adversarial_loss_value = params.adv_coeff * cross_entropy_loss(fake_probs, real_labels)
            total_var_loss = params.tv_loss_coeff * total_variance_loss_calc(params.vgg_rescale_coeff * (high_res_feature - super_res_feature)**2)
            
            total_gen_loss = actual_perceptual_loss + adversarial_loss_value + total_var_loss + l2_loss_value
            
            gen_optimizer.zero_grad()
            discrim_optimizer.zero_grad()
            total_gen_loss.backward()
            gen_optimizer.step()

        post_training_epoch += 1
        if post_training_epoch % 2 == 0:
            # print results every even epoch
            print(post_training_epoch)
            print(total_gen_loss.item())
            print(total_d_loss.item())
            print('=========')

        if post_training_epoch % 500 ==0:
            # save model checkpoint every 500 iterations
            torch.save(superres_gen.state_dict(), './model/SuperResNet_gen_%03d.pt'%post_training_epoch)
            torch.save(superres_discriminator.state_dict(), './model/SuperResNet_disc_%03d.pt'%post_training_epoch)
