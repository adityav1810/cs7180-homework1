'''
CS 7180 : Advanced Perception 
Homework 1

Authors : Luv Verma and Aditya Varshney
This file contains the loss functions for the model
'''

import torch
import torch.nn as nn
from torchvision import transforms


class RGBMeanShift(nn.Conv2d):
    '''
    Compute Mean Shift of a Conv2D layer
    '''
    def __init__(
        self, rgb_range_val=1,
        normalization_mean=(0.485, 0.456, 0.406), normalization_std=(0.229, 0.224, 0.225), operation_sign=-1):

        super(RGBMeanShift, self).__init__(3, 3, kernel_size=1)
        std_tensor = torch.Tensor(normalization_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std_tensor.view(3, 1, 1, 1)
        self.bias.data = operation_sign * rgb_range_val * torch.Tensor(normalization_mean) / std_tensor
        self.device_instance = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for param in self.parameters():
            param.requires_grad = False


class VGGPerceptualLoss(nn.Module):
    '''
    Compute Perceptual loss from VGG pretrained model
    '''
    def __init__(self, vgg_model):
        super(VGGPerceptualLoss, self).__init__()
        self.norm_mean_values = [0.485, 0.456, 0.406]
        self.norm_std_values = [0.229, 0.224, 0.225]
        self.device_instance = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean_shift = RGBMeanShift(norm_mean=self.norm_mean_values, norm_std=self.norm_std_values).to(self.device_instance)
        self.vgg_instance = vgg_model
        self.loss_criterion = nn.MSELoss()
    def forward(self, hr_image, sr_image, layer_name='relu5_4'):
        ## HR and SR should be normalized [0,1]
        normalized_hr = self.mean_shift(hr_image)
        normalized_sr = self.mean_shift(sr_image)
        
        hr_feature = getattr(self.vgg_instance(normalized_hr), layer_name)
        sr_feature = getattr(self.vgg_instance(normalized_sr), layer_name)
        
        return self.loss_criterion(hr_feature, sr_feature), hr_feature, sr_feature

class TotalVariationLoss(nn.Module):
    '''
    Compute Total Variance Loss
    '''
    def __init__(self, tv_weight=1):
        super(TotalVariationLoss, self).__init__()
        self.tv_weight_value = tv_weight

    def forward(self, image_tensor):
        batch_count = image_tensor.size()[0]
        height = image_tensor.size()[2]
        width = image_tensor.size()[3]
        height_count = self.tensor_dimension(image_tensor[:, :, 1:, :])
        width_count = self.tensor_dimension(image_tensor[:, :, :, 1:])
        height_variation = torch.pow((image_tensor[:, :, 1:, :] - image_tensor[:, :, :height - 1, :]), 2).sum()
        width_variation = torch.pow((image_tensor[:, :, :, 1:] - image_tensor[:, :, :, :width - 1]), 2).sum()
        
        return self.tv_weight_value * 2 * (height_variation / height_count + width_variation / width_count) / batch_count

    @staticmethod
    def tensor_dimension(tensor_obj):
        return tensor_obj.size()[1] * tensor_obj.size()[2] * tensor_obj.size()[3]
