'''
CS 7180 : Advanced Perception 
Homework 1
Authors : Luv Verma and Aditya Varshney

This file contains the main argument parser which stimulates the model training, testing, and validation tasks.
'''

from operation_mode import *
import argparse

arg_parser = argparse.ArgumentParser()

def string_to_boolean(value):
    return value.lower() in ('true')

arg_parser.add_argument("--LR_path", type=str, default='../dataSet/DIV2K/DIV2K_valid_LR_bicubic/X4')
arg_parser.add_argument("--GT_path", type=str, default='../dataSet/DIV2K/DIV2K_train_HR/')
arg_parser.add_argument("--res_num", type=int, default=16)
arg_parser.add_argument("--num_workers", type=int, default=0)
arg_parser.add_argument("--batch_size", type=int, default=16)
arg_parser.add_argument("--L2_coeff", type=float, default=1.0)
arg_parser.add_argument("--adv_coeff", type=float, default=1e-3)
arg_parser.add_argument("--tv_loss_coeff", type=float, default=0.0)
arg_parser.add_argument("--pre_train_epoch", type=int, default=8000)
arg_parser.add_argument("--fine_train_epoch", type=int, default=4000)
arg_parser.add_argument("--scale", type=int, default=4)
arg_parser.add_argument("--patch_size", type=int, default=24)
arg_parser.add_argument("--feat_layer", type=str, default='relu5_4')
arg_parser.add_argument("--vgg_rescale_coeff", type=float, default=0.006)
arg_parser.add_argument("--fine_tuning", type=string_to_boolean, default=False)
arg_parser.add_argument("--in_memory", type=string_to_boolean, default=True)
arg_parser.add_argument("--generator_path", type=str)
arg_parser.add_argument("--operation_mode", type=str, default='train')

parsed_args = arg_parser.parse_args()

if parsed_args.operation_mode == 'train':
    train(parsed_args)
    
elif parsed_args.operation_mode == 'test':
    test(parsed_args)
    
elif parsed_args.operation_mode == 'validate':
    validate(parsed_args)
