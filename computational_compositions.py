import numpy as np
from utils import apply_sr_image, convert_numpy_255_to_tensor_zero_one, convert_tensor_zero_one_to_numpy_255, make_video_mp4
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
import torch
import pickle

import torch
from numpy.lib.shape_base import row_stack
from tiling.tile_system import TileGridSystem

def run_expansion(grid_sys, expansion_idx, plot_losses, bw, opp):
    content = False
    print("Expanding to index:", expansion_idx)
    while not content:
        
        grid_sys.expand(idx=expansion_idx, opposite_transform=opp, reduce_black_and_white=bw, plot_losses=plot_losses, log_every=100, plot_grid_after=True)
        print()

        content_string = input("Do you want to continue? (y/n) ")
        
        if content_string == "y":
            content = True

    return grid_sys

def get_row_col_input(nrows, ncols):
    valid_row_col = False
    while not valid_row_col:
        row_str = input("What should be the row of the new tile?")
        col_str = input("What should be the col of the new tile?")
        
        row, col = int(row_str), int(col_str)

        if row < nrows and row >= 0 and col < ncols and col >= 0:
            valid_row_col = True
    return row, col

def run_interactive_comp_comp(netG, initial_latent=None, nrows=4, ncols=4, n_pixels_boundary=4,
                              plot_losses=False, device="cuda:0",
                              nz=100, lr=0.05, num_steps=400, latent_range=2.5,
                              max_weight_pixel=100.0, min_weight_pixel=99.0, max_weight_reg=5.0, min_weight_reg=3.0,
                              latent_std=1.0, initial_noise_factor=0.05, noise_ramp_length=0.75, max_noise=0.05):
    
    netG = netG.to(device).eval()

    # if initial_latent is None:
    #     initial_latent = torch.randn((1, nz, 1, 1)).to(device)

    grid_sys = TileGridSystem(netG=netG, ncols=ncols, nrows=nrows, nz=nz, device=device, lr=lr, num_steps=num_steps, 
                            latent_range=latent_range, n_pixels_boundary=n_pixels_boundary, initial_latent=initial_latent,
                            max_weight_pixel=max_weight_pixel, min_weight_pixel=min_weight_pixel, max_weight_reg=max_weight_reg, 
                            min_weight_reg=min_weight_reg, latent_std=latent_std, initial_noise_factor=initial_noise_factor, 
                            noise_ramp_length=noise_ramp_length, max_noise=max_noise)
    
    while not grid_sys.check_full():
        row, col = get_row_col_input(nrows, ncols)

        bw = True if input("Use BW mode? (y/n)") == "y" else False
        opp = True if input("Use Opposite mode? (y/n)") == "y" else False
        
        idx = grid_sys.row_col_to_idx(row, col)
        if idx in grid_sys.optimisations:
            print(f"Row = {row}, col = {col} already existed, so erasing it.")
            grid_sys.erase_idx(idx)
        
        grid_sys = run_expansion(grid_sys=grid_sys, expansion_idx=idx, plot_losses=plot_losses, bw=bw, opp=opp)

    return grid_sys

def save_grid_sys_as_pickle(grid_sys, save_dir):
    pickle_dict = dict()
    pickle_dict["imgs"] = {k:v.detach().cpu() for k, v in grid_sys.imgs.items()}
    pickle_dict["optimisations"] = dict()
    for tile_idx, v in grid_sys.optimisations.items():
        if torch.is_tensor(v) or v is None:
            pickle_dict["optimisations"][tile_idx] = v
        else:
            pickle_dict["optimisations"][tile_idx] = [elem.detach().cpu() for elem in v]

    pickle_path = f"{save_dir}/grid_sys.pickle"
    pickle.dump( pickle_dict, open(pickle_path, "wb"))
    print("Dumped as pickle in", pickle_path)
    return pickle_dict

def export_optimisation_video_and_final_grid_image(grid_sys_dict, ncols, nrows, num_optim_steps, save_dir, lap_sr_dir, super_resolution=False, super_resolution_times=4):
    n_imgs = ncols * nrows
    im_shape = grid_sys_dict["imgs"][0].shape

    print("Number of images (nrows x ncols):", f"{n_imgs} ({nrows} x {ncols}")
    print("Image shape:", im_shape)
    print("Number of optimisation steps:", num_optim_steps)

    # Fill the sampled static base image to match length of other optimisations
    for im_idx in range(n_imgs):
        if grid_sys_dict["optimisations"][im_idx] is None:
            grid_sys_dict["optimisations"][im_idx] = [grid_sys_dict["imgs"][im_idx] for _ in range(num_optim_steps)]

    # Video path
    if super_resolution:
        video_path = f'{save_dir}/grid-optimisation-video-x{super_resolution_times}.mp4'
        final_grid_image_path =  f'{save_dir}/optimised-grid-image-x{super_resolution_times}.png'
    else:
        video_path = f'{save_dir}/grid-optimisation-video.mp4'
    
    final_grid_image_path =  f'{save_dir}/optimised-grid-image.png'

    # Make video
    print(f"Start exporting video with super_resolution = {super_resolution}")
    video_images = []
    for step in range(num_optim_steps):
        if step % 10 == 0:
            print(f"{step}/{num_optim_steps}", end="\r")

        if not super_resolution:
            ims_grid = np.stack([grid_sys_dict["optimisations"][im_idx][step] for im_idx in range(n_imgs)])
            ims_grid_tensor_batch = torch.Tensor(ims_grid).float()
        
        # Apply super resolution to scale up
        else:
            ims_grid = []
            for im_idx in range(n_imgs):
                # [c, h, w] -> [h, w, c]
                img_rgb = convert_tensor_zero_one_to_numpy_255(grid_sys_dict["optimisations"][im_idx][step])
                # [h, w, c] -> [c, h, w]
                img_rgb_scaled = apply_sr_image(img_rgb, times=super_resolution_times, lap_sr_dir=lap_sr_dir)
                ims_grid.append(img_rgb_scaled)
                ims_grid_tensor_batch = convert_numpy_255_to_tensor_zero_one(np.stack(ims_grid))
        
        im_tensor_grid = make_grid(ims_grid_tensor_batch, pad_value=1.0, value_range=(0.0, 1.0), nrow=ncols, padding=0)

        if step == 0 or step == num_optim_steps - 1:
            plt.imshow(im_tensor_grid.permute(1, 2, 0))
            plt.axis('off')
            plt.show()

        im_grid_export = convert_tensor_zero_one_to_numpy_255(im_tensor_grid)
        video_images.append(im_grid_export)

    make_video_mp4(video_images, video_path=video_path, multiply_frames=6)
    Image.fromarray(video_images[-1]).save(final_grid_image_path)

    print("Done")