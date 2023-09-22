'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the building blocks of the NN
defines the convolutional block, the Residual Block, upsampling block, and discriminator
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvHelper(nn.Conv2d):
    '''
    Helper for the convolutional block
    '''
    def __init__(self, input_channels, output_channels, filter_size, stride_val, padding_val, use_bias):
        super(ConvHelper, self).__init__(input_channels, output_channels, filter_size, stride_val, padding_val, bias=use_bias)
        
        self.weight.data = torch.normal(torch.zeros((output_channels, input_channels, filter_size, filter_size)), 0.02)
        self.bias.data = torch.zeros((output_channels))
        
        for param in self.parameters():
            param.requires_grad = True

class ConvolutionBlock(nn.Module):
    '''
    Main Convolutional Block
    '''
    def __init__(self, in_chan, out_chan, filter_size, use_bn=False, activation_fn=None, stride=1, use_bias=True):
        super(ConvolutionBlock, self).__init__()
        layers = []
        layers.append(ConvHelper(in_chan, out_chan, filter_size, stride, filter_size // 2, use_bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_chan))
        if activation_fn:
            layers.append(activation_fn)
        self.module_seq = nn.Sequential(*layers)
        
    def forward(self, tensor_input):
        return self.module_seq(tensor_input)
        
class ResidualBlock(nn.Module):
    '''
    Residual Block for the network
    '''
    def __init__(self, chan_count, filter_size, activation_fn=nn.ReLU(inplace=True)):
        super(ResidualBlock, self).__init__()
        blocks = []
        blocks.append(ConvolutionBlock(chan_count, chan_count, filter_size, use_bn=True, activation_fn=activation_fn))
        blocks.append(ConvolutionBlock(chan_count, chan_count, filter_size, use_bn=True))
        self.residual_seq = nn.Sequential(*blocks)
        
    def forward(self, tensor_input):
        residual = self.residual_seq(tensor_input)
        return residual + tensor_input
    
class BaseBlock(nn.Module):
    def __init__(self, input_channels, output_channels, filter_size, num_residuals, activation_fn=nn.ReLU(inplace=True)):
        super(BaseBlock, self).__init__()
        block_layers = []
        self.init_conv = ConvolutionBlock(input_channels, output_channels, filter_size, activation_fn=activation_fn)
        for _ in range(num_residuals):
            block_layers.append(ResidualBlock(output_channels, filter_size, activation_fn))
        block_layers.append(ConvolutionBlock(output_channels, output_channels, filter_size, use_bn=True))
        self.block_seq = nn.Sequential(*block_layers)
        
    def forward(self, tensor_input):
        initial = self.init_conv(tensor_input)
        block_output = self.block_seq(initial)
        return block_output + initial
        
class UpscaleBlock(nn.Module):
    '''
    Upsampling Block
    '''
    def __init__(self, channels, filter_size, scale_factor, activation_fn=nn.ReLU(inplace=True)):
        super(UpscaleBlock, self).__init__()
        layers = []
        layers.append(ConvolutionBlock(channels, channels * scale_factor * scale_factor, filter_size))
        layers.append(nn.PixelShuffle(scale_factor))
        if activation_fn:
            layers.append(activation_fn)
        self.up_seq = nn.Sequential(*layers)
    def forward(self, tensor_input):
        return self.up_seq(tensor_input)

class DiscriminatorBlock(nn.Module):
    '''
    Block for Discriminator Network
    '''
    def __init__(self, input_features, output_features, filter_size, activation_fn=nn.LeakyReLU(inplace=True)):
        super(DiscriminatorBlock, self).__init__()
        blocks = []
        blocks.append(ConvolutionBlock(input_features, output_features, filter_size, use_bn=True, activation_fn=activation_fn))
        blocks.append(ConvolutionBlock(output_features, output_features, filter_size, use_bn=True, activation_fn=activation_fn, stride=2))
        self.discrim_seq = nn.Sequential(*blocks)
    def forward(self, tensor_input):
        return self.discrim_seq(tensor_input)
