
import torch
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def save_LR_HR_SR_images(args):
    '''
    Helper method for saving results.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = mydata(GT_path=args.GT_path, LR_path=args.LR_path, in_memory=False, transform=None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    
    generator = Generator(img_feat=3, n_feats=64, kernel_size=3, num_block=args.res_num)
    generator.load_state_dict(torch.load(args.generator_path))
    generator = generator.to(device)
    generator.eval()

    # Define paths for saving images
    lr_save_path = './result/LR_Images/'
    gt_save_path = './result/GT_Images/'
    sr_save_path = './result/Set14_Test/'

    # Ensure directories exist
    for path in [lr_save_path, gt_save_path, sr_save_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    with torch.no_grad():
        for i, te_data in enumerate(loader):
            gt = te_data['GT'].to(device)
            lr = te_data['LR'].to(device)
            
            # Save LR and HR images
            Image.fromarray((lr[0].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)).save(os.path.join(lr_save_path, f"LR_res_{i:04d}.png"))
            Image.fromarray((gt[0].cpu().numpy().transpose(1, 2, 0) * 127.5 + 127.5).astype(np.uint8)).save(os.path.join(gt_save_path, f"HR_res_{i:04d}.png"))
            
            # Generate and save SR image
            output, _ = generator(lr)
            output = output[0].cpu().numpy().transpose(1, 2, 0)
            output = (output * 127.5 + 127.5).astype(np.uint8)
            Image.fromarray(output).save(os.path.join(sr_save_path, f"SR_res_{i:04d}.png"))

            


def plot_images_side_by_side():
    '''
    Another method to present images side by side
    '''
    sr_image_dir = './result/Set14_Test/'
    n_images = len([name for name in os.listdir(sr_image_dir) if os.path.isfile(os.path.join(sr_image_dir, name))])

    for i in range(n_images):
        lr_image = Image.open(f'./result/LR_Images/LR_res_{i:04d}.png')
        hr_image = Image.open(f'./result/GT_Images/HR_res_{i:04d}.png')
        sr_image = Image.open(f'./result/Set14_Test_490_resized/res_{i:04d}.png')

        # Convert images to numpy arrays and normalize them for PSNR calculation
        y_output = np.array(sr_image) / 255.0
        y_gt = np.array(hr_image) / 255.0
        y_low_res = np.array(lr_image.resize(hr_image.size, Image.BICUBIC)) / 255.0

        if y_gt.shape[1] != y_output.shape[1] * args.scale:
            y_output_img = Image.fromarray((y_output * 255.0).astype(np.uint8))
            y_output_resized = np.array(y_output_img.resize((y_gt.shape[1], y_gt.shape[0]), Image.BICUBIC)) / 255.0
        else:
            crop_size = args.scale
            y_output_resized = y_output[crop_size:-crop_size, crop_size:-crop_size]
            y_gt = y_gt[crop_size:-crop_size, crop_size:-crop_size]
            y_low_res = y_low_res[crop_size:-crop_size, crop_size:-crop_size]

        psnr_value = compare_psnr(y_output_resized, y_gt, data_range=1.0)
        psnr_value_lr_hr = compare_psnr(y_low_res, y_gt, data_range=1.0)  # Since y_low_res has been upscaled
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(lr_image)
        plt.title(f"Low Resolution\nSize: {lr_image.size}\nPixels: {lr_image.size[0]*lr_image.size[1]}\nPSNR: {psnr_value_lr_hr:.2f} dB")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(sr_image)
        plt.title(f"Super-Resolved (Output)\nSize: {sr_image.size}\nPixels: {sr_image.size[0]*sr_image.size[1]}\nPSNR: {psnr_value:.2f} dB")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(hr_image)
        plt.title(f"Ground Truth\nSize: {hr_image.size}\nPixels: {hr_image.size[0]*hr_image.size[1]}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
# save_LR_HR_SR_images(args)
plot_images_side_by_side()


class ResizeImages:
    def __init__(self, source_folder, target_folder, target_size):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.target_size = target_size
    
    def process(self):
        # Create the target folder if it doesn't exist
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)
        
        # Traverse through all files in the source folder
        for img_file in os.listdir(self.source_folder):
            # Construct the full file path
            img_path = os.path.join(self.source_folder, img_file)
            
            # Open the image and resize it
            try:
                img = Image.open(img_path)
                img_resized = img.resize(self.target_size, Image.LANCZOS)
                
                # Save the resized image in the target folder
                img_resized.save(os.path.join(self.target_folder, img_file))
            except Exception as e:
                print(f"Error processing {img_file}: {e}")


# Resize low-resolution images
lr_target_size = (490,490)
resizer_LR = ResizeImages(args.LR_path, 'dataSet/DIV2K/LR_Set14_resized_490', lr_target_size)
resizer_LR.process()

# Resize high-resolution images
gt_target_size = (490,490)
resizer_GT = ResizeImages(args.GT_path, 'dataSet/DIV2K/HR_Set14_resized_490', gt_target_size)
resizer_GT.process()