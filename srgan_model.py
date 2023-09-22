'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file constructs the SRGAN network
'''

import torch
import torch.nn as nn
from ops import *

class SuperResGenerator(nn.Module):
    '''
    Define the Generator Network for the SRGAN
    '''
    def __init__(self, input_channels=3, feature_channels=64, filter_size=3, num_res_blocks=16, activation_fn=nn.PReLU(), upscale_factor=4):
        super(SuperResGenerator, self).__init__()
        # Initial convolutional layer
        self.initial_conv = ConvolutionBlock(input_channels, feature_channels, filter_size=9, activation_fn=activation_fn)
        
        # Residual blocks
        residual_blocks = [ResidualBlock(feature_channels, filter_size, activation_fn) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*residual_blocks)
        
        # Middle convolutional layer
        self.middle_conv = ConvolutionBlock(feature_channels, feature_channels, filter_size, use_bn=True)
        
        # Upsampling blocks
        if upscale_factor == 4:
            upsample_layers = [UpscaleBlock(feature_channels, filter_size, scale=2, activation_fn=activation_fn) for _ in range(2)]
        else:
            upsample_layers = [UpscaleBlock(feature_channels, filter_size, scale=upscale_factor, activation_fn=activation_fn)]
            
        self.upsample_seq = nn.Sequential(*upsample_layers)
        
        # Final convolutional layer
        self.final_conv = ConvolutionBlock(feature_channels, input_channels, filter_size, activation_fn=nn.Tanh())
        
    def forward(self, tensor_input):
        tensor_input = self.initial_conv(tensor_input)
        skip_tensor = tensor_input
        
        tensor_input = self.res_blocks(tensor_input)
        tensor_input = self.middle_conv(tensor_input)
        # Add a skip connection
        features = tensor_input + skip_tensor
        
        tensor_input = self.upsample_seq(features)
        tensor_input = self.final_conv(tensor_input)
        
        return tensor_input, features
    
class SuperResDiscriminator(nn.Module):
    '''
    Define the Discriminator Network for the SRGAN 
    '''
    def __init__(self, input_channels=3, feature_channels=64, filter_size=3, activation_fn=nn.LeakyReLU(inplace=True), num_disc_blocks=3, patch_dim=96):
        super(SuperResDiscriminator, self).__init__()
        self.activation_fn = activation_fn
        
        # Initial layers
        self.start_conv1 = ConvolutionBlock(input_channels, feature_channels, filter_size, activation_fn=self.activation_fn)
        self.start_conv2 = ConvolutionBlock(feature_channels, feature_channels, filter_size, stride=2, activation_fn=self.activation_fn)
        
        # Discriminator blocks
        disc_blocks = [DiscriminatorBlock(feature_channels * (2 ** i), feature_channels * (2 ** (i + 1)), filter_size, activation_fn=self.activation_fn) for i in range(num_disc_blocks)]
        self.disc_seq = nn.Sequential(*disc_blocks)
        
        # Linear layers
        self.linear_dim = ((patch_dim // (2 ** (num_disc_blocks + 1))) ** 2) * (feature_channels * (2 ** num_disc_blocks))
        
        end_layers = [
            nn.Linear(self.linear_dim, 1024),
            self.activation_fn,
            nn.Linear(1024, 1),
            nn.Sigmoid()
        ]
        self.end_seq = nn.Sequential(*end_layers)
        
    def forward(self, tensor_input):        
        tensor_input = self.start_conv2(tensor_input)
        tensor_input = self.disc_seq(tensor_input)        
        tensor_input = tensor_input.view(-1, self.linear_dim)
        tensor_input = self.end_seq(tensor_input)
        return tensor_input
