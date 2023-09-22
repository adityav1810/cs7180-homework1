'''
CS 7180 : Advanced Perception 
Homework 1

Authors : Luv Verma and Aditya Varshney
This file contains the dataset creator and dataloader
'''

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random
import matplotlib.pyplot as plt

class CustomData(Dataset):
    '''
        Custom DataLoader for training purposes.
        Loads images from lr_dir and hr_dir.
        Converts them into list of numpy images.
    '''
    
    def __init__(self, lr_dir, hr_dir, cache_in_memory=True, data_transform=None):
        '''
        Constructor to load entire data
        '''
        
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.cache_in_memory = cache_in_memory
        self.data_transform = data_transform
        
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_files = sorted(os.listdir(hr_dir))
        
        if cache_in_memory:
            self.lr_files = [np.array(Image.open(os.path.join(self.lr_dir, img_path)).convert("RGB")).astype(np.uint8) for img_path in self.lr_files]
            self.hr_files = [np.array(Image.open(os.path.join(self.hr_dir, img_path)).convert("RGB")).astype(np.uint8) for img_path in self.hr_files]
        
    def __len__(self):
        '''
        returns size of dataset
        '''
        
        return len(self.lr_files)
        
    def __getitem__(self, idx):
        '''
        returns one image from dataset
        '''
        
        img_data = {}
        
        if self.cache_in_memory:
            hr_img = self.hr_files[idx].astype(np.float32)
            lr_img = self.lr_files[idx].astype(np.float32)
            
        else:
            hr_img = np.array(Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert("RGB"))
            lr_img = np.array(Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert("RGB"))

        img_data['hr_img'] = (hr_img / 127.5) - 1.0
        img_data['lr_img'] = (lr_img / 127.5) - 1.0
                
        if self.data_transform is not None:
            img_data = self.data_transform(img_data)
            
        img_data['hr_img'] = img_data['hr_img'].transpose(2, 0, 1).astype(np.float32)
        img_data['lr_img'] = img_data['lr_img'].transpose(2, 0, 1).astype(np.float32)
        img_name = self.lr_files[idx] if self.cache_in_memory else os.path.basename(os.path.join(self.lr_dir, self.lr_files[idx]))
        img_data['img_name'] = img_name

        return img_data
    
    
class TestDataset(Dataset):
    '''
    Custom Dataloader for testing purposes.
    
    '''
    def __init__(self, lr_dir, cache_in_memory=True, data_transform=None):
        
        self.lr_dir = lr_dir
        self.lr_files = sorted(os.listdir(lr_dir))
        self.cache_in_memory = cache_in_memory
        if cache_in_memory:
            self.lr_files = [np.array(Image.open(os.path.join(self.lr_dir, img_path))) for img_path in self.lr_files]
        
    def __len__(self):
        
        return len(self.lr_files)
        
    def __getitem__(self, idx):
        
        img_data = {}
        
        if self.cache_in_memory:
            lr_img = self.lr_files[idx]
            
        else:
            lr_img = np.array(Image.open(os.path.join(self.lr_dir, self.lr_files[idx])))

        img_data['lr_img'] = (lr_img / 127.5) - 1.0                
        img_data['lr_img'] = img_data['lr_img'].transpose(2, 0, 1).astype(np.float32)
        
        return img_data


class ImageCrop(object):
    '''
    Preprocessing class.
    Class to crop a part of an image.
    
    '''
    def __init__(self, scaling_factor, patch_dimension):
        
        self.scaling_factor = scaling_factor
        self.patch_dimension = patch_dimension
        
    def __call__(self, img_sample):
        '''
        crop image using patch_dimension and scaling factor
        '''
        lr_img, hr_img = img_sample['lr_img'], img_sample['hr_img']
        img_h, img_w = lr_img.shape[:2]
        
        img_x = random.randrange(0, img_w - self.patch_dimension +1)
        img_y = random.randrange(0, img_h - self.patch_dimension +1)
        
        scaled_x = img_x * self.scaling_factor
        scaled_y = img_y * self.scaling_factor
        
        lr_patch = lr_img[img_y : img_y + self.patch_dimension, img_x : img_x + self.patch_dimension]
        hr_patch = hr_img[scaled_y : scaled_y + (self.scaling_factor * self.patch_dimension), scaled_x : scaled_x + (self.scaling_factor * self.patch_dimension)]
        
        return {'lr_img' : lr_patch, 'hr_img' : hr_patch}

class ImageFlip(object):
    '''
    Preprocessing class.
    Class to flip an image.
    '''
    
    def __call__(self, img_sample):
        '''
        Decide whether to rotate and/or flip an image horizontally or vertically.
        Perform the operation accordingly.
        '''
        lr_img, hr_img = img_sample['lr_img'], img_sample['hr_img']
        
        horizontal_flip = random.randrange(0,2)
        vertical_flip = random.randrange(0,2)
        rotate_flag = random.randrange(0,2)
    
        if horizontal_flip:
            # horizontal flip             
            lr_img = np.fliplr(lr_img)
            hr_img = np.fliplr(hr_img)
        
        if vertical_flip:
            # vertical flip
            lr_img = np.flipud(lr_img)
            hr_img = np.flipud(hr_img)
            
        if rotate_flag:
            # rotate image             
            lr_img = lr_img.transpose(1, 0, 2)
            hr_img = hr_img.transpose(1, 0, 2)
        
        return {'lr_img' : lr_img, 'hr_img' : hr_img}
