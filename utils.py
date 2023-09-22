import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def store_images(params):
    '''
    Helper function to save the images.
    '''
    device_config = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_set = mydata(GT_path=params.GT_path, LR_path=params.LR_path, in_memory=False, transform=None)
    data_loader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=params.num_workers)
    
    superres_gen = SuperResGenerator(input_channels=3, feature_channels=64, filter_size=3, num_res_blocks=params.res_num)
    superres_gen.load_state_dict(torch.load(params.generator_path))
    superres_gen = superres_gen.to(device_config)
    superres_gen.eval()

    # Paths for saving images
    low_res_path = './result/LR_Images/'
    gt_res_path = './result/GT_Images/'
    super_res_path = './result/Set14_Test/'

    # Make sure directories are present
    for path in [low_res_path, gt_res_path, super_res_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    with torch.no_grad():
        for idx, data_point in enumerate(data_loader):
            ground_truth = data_point['GT'].to(device_config)
            low_res = data_point['LR'].to(device_config)
            
            # Save LR and GT images
            Image.fromarray((low_res[0].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)).save(os.path.join(low_res_path, f"LR_res_{idx:04d}.png"))
            Image.fromarray((ground_truth[0].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)).save(os.path.join(gt_res_path, f"HR_res_{idx:04d}.png"))
            
            # Generate and save super-resolved image
            output, _ = superres_gen(low_res)
            output_img = output[0].cpu().numpy().transpose(1, 2, 0)
            output_img = (output_img * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(output_img).save(os.path.join(super_res_path, f"SR_res_{idx:04d}.png"))


class ImageResizer:
    def __init__(self, src_dir, dest_dir, dest_size):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        self.dest_size = dest_size
    
    def execute(self):
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)
        
        for image_name in os.listdir(self.src_dir):
            image_path = os.path.join(self.src_dir, image_name)
            try:
                img = Image.open(image_path)
                resized_img = img.resize(self.dest_size, Image.LANCZOS)
                resized_img.save(os.path.join(self.dest_dir, image_name))
            except Exception as err:
                print(f"Error processing {image_name}: {err}")


# Resize the images
lr_dim = (490,490)
lr_resizer = ImageResizer(params.LR_path, 'dataSet/DIV2K/LR_Set14_resized_490', lr_dim)
lr_resizer.execute()

gt_dim = (490,490)
gt_resizer = ImageResizer(params.GT_path, 'dataSet/DIV2K/HR_Set14_resized_490', gt_dim)
gt_resizer.execute()
