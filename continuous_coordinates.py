import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision.utils import make_grid
from utils import apply_sr_image, right_rotate, make_video_mp4, convert_numpy_255_to_tensor_zero_one, convert_tensor_zero_one_to_numpy_255

def interpolate(tensor_1, tensor_2, num_steps=10):
    assert tensor_1.shape == tensor_2.shape, "tensors must have same shape"
    step = (tensor_2 - tensor_1) / num_steps
    result = [tensor_1 + step * i for i in range(num_steps + 1)]
    assert torch.all(
        torch.isclose(result[-1], tensor_2, rtol=1e-05, atol=1e-05)), "did not pass interpolate sanity check"
    return torch.stack(result)

def interpolation_chain(latent_tensors, num_steps):
    z_from = latent_tensors[0]
    interpolations = []
    for i in range(1, len(latent_tensors)):
        z_to = latent_tensors[i]
        interpolations.append(interpolate(z_from, z_to, num_steps=num_steps))
        z_from = z_to
    return torch.cat(interpolations)

def make_interpolation_grid(netG, n_rows, n_cols, save_dir, interpolation_noise=None, pad_value=1.0, padding=12, 
                            grid_img_size=100, nz=100, device="cpu", upscale=False, upscale_times=4, 
                            upscale_models_dir="lap_sr_models", plot_grid=True):
    # Set generator network to correct device and in evaluation mode
    netG = netG.to(device).eval()

    # Make directories to save the results to
    os.makedirs(f"{save_dir}")
    os.makedirs(f"{save_dir}/sep_imgs")
    
    # If no interpolation is given, sample some randomly
    if interpolation_noise is not None:
        assert interpolation_noise.dim() == 4, f"interpolation noise needs to be 4D [n_rows, nz, 1, 1], current shape {interpolation_noise.shape}"
        n_rows = n_cols = len(interpolation_noise) + 1
        print(f"N rows (= N cols) is set to the number of interpolation noise points (= {n_rows})")
    else:    
        interpolation_noise = torch.randn(size=(n_rows - 1, nz, 1, 1)).to(device)

    # Repeat the last latent at the end
    interpolation_noise = torch.cat([interpolation_noise, interpolation_noise[0].unsqueeze(0)])

    # Interpolate the latents
    interpolated_zs = interpolation_chain(latent_tensors=interpolation_noise, num_steps=n_cols).to(device)
    
    # Forward the latents to generate images
    with torch.no_grad():
        interpolated_imgs = netG(interpolated_zs).cpu()
        interpolated_imgs = torch.cat([interpolated_imgs, interpolated_imgs[0].unsqueeze(0)])
    
    # Upscale the images from 64x64 to something large (x4 or x8)
    if upscale:
        imgs_upscaled = []
        # The super resolution model expects uint8 images 0-255 range
        img_np_arrays = convert_tensor_zero_one_to_numpy_255(interpolated_imgs)
        for im_idx in range(len(interpolated_imgs)):
            im_array_scaled = apply_sr_image(img_np_arrays[im_idx], times=upscale_times, lap_sr_dir=upscale_models_dir)
            imgs_upscaled.append(im_array_scaled)
        interpolated_imgs = convert_numpy_255_to_tensor_zero_one(np.stack(imgs_upscaled))

    # Save separate images
    print("Saving separate interpolated images")
    img_np_arrays = convert_tensor_zero_one_to_numpy_255(interpolated_imgs)
    for im_idx in range(len(interpolated_imgs)):
        Image.fromarray(img_np_arrays[im_idx]).save(f"{save_dir}/sep_imgs/{im_idx}.png")

    # Convert images to correct size for the grid
    img_np_arrays_resized = []
    for im_idx in range(len(interpolated_imgs)):
        # a bit cumbersome to open again, but to resize as PIL Image object
        pil_image = Image.open(f"{save_dir}/sep_imgs/{im_idx}.png")
        pil_image = pil_image.resize((grid_img_size, grid_img_size)) # large = 256, small = 100
        img_np_arrays_resized.append(np.array(pil_image))

    # Make animation
    print("Making animation video of interpolation.")
    video_path_animation = f'{save_dir}/animation.mp4'
    make_video_mp4(img_np_arrays, video_path_animation, multiply_frames=6)

    # Make rotating grid animation video
    print("Making grid animation video.")
    video_path_grid_animation = f'{save_dir}/grid-animation.mp4'
    n_cycle_steps = len(interpolated_imgs)
    grid_np_imgs = []
    for cycle_idx in range(n_cycle_steps):
        print(f"{cycle_idx:3d}/{n_cycle_steps}", end="\r")
        indices = np.array(right_rotate(list(np.arange(n_cycle_steps, dtype=int)), cycle_idx))
        im_batch = np.array(img_np_arrays_resized)[indices]
        im_batch_tensor = convert_numpy_255_to_tensor_zero_one(im_batch)
        im_tensor_grid = make_grid(im_batch_tensor, pad_value=pad_value, value_range=(0.0, 1.0), nrow=n_cols, padding=padding)
        grid_np_imgs.append(convert_tensor_zero_one_to_numpy_255(im_tensor_grid))
    make_video_mp4(grid_np_imgs, video_path_grid_animation, multiply_frames=6)
    
    # Save static grid image
    print("Saving static grid image")
    Image.fromarray(grid_np_imgs[0]).save(f"{save_dir}/grid.png")

    print("Done!")

    if plot_grid:
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.imshow(grid_np_imgs[0])
        plt.axis("off")
        plt.show()