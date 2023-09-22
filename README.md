CS7180 : Advanced Perception 
Homework 1 : Image Enhancement using SRGAN
Authors : Luv Verma and Aditya Varshney
----------------------

This repository holds the entire codebase of our submission for Assigment 1.
Please refer to the pdf file for a full project report summary



---------------------------
Instructions to run the code 
1. setup pytorch
2. use the terminal to run the model in train, test or validate mode

 train the model
-------------------------

python main.py --LR_path ./LR_imgs_dir --GT_path ./GT_imgs_dir


test
-----------------------
python main.py --mode test --LR_path ./LR_imgs_dir --GT_path ./GT_imgs_dir --generator_path ./model/SRGAN.pt


validate
---------------------------
python main.py --mode test_only --LR_path ./LR_imgs_dir --generator_path ./model/SRGAN.pt



