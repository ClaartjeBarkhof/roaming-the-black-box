import os
import cv2
import shutil

import torch
from torch.distributions import Normal
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

import imageio

def make_image_grid(images, ncols, cell_size=1.5, un_normalize=False, set_idx_title=False, border=False, wspace=0.0,
                    hspace=0.0, title=None, save_as=None, transparent=True, dpi=150, show_plot=True, title_y=None):
    assert images.dim() == 4, "we expect a 4D tensor for a batch of images"

    # if channel first, switch to channel last
    if images.shape[1] < 4:
        images = images.permute(0, 2, 3, 1)

    nimgs = len(images)
    nrows = int(np.ceil(nimgs / ncols))

    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * cell_size, nrows * cell_size))
    
    if ncols == 1:
        axs = axs.reshape(-1, 1)
    
    if nrows == 1:
        axs = axs.reshape(1, -1)

    for i in range(nimgs):
        r = i // ncols
        c = i % ncols

        im_i = images[i]

        if un_normalize:
            im_i = (im_i + 1.0) / 2.0

        axs[r, c].imshow(im_i, vmin=0.0, vmax=1.0, aspect="auto")

        if border:
            axs[r, c].set_xticks([])
            axs[r, c].set_yticks([])
        else:
            axs[r, c].axis('off')

        if set_idx_title:
            axs[r, c].set_title(str(i))

    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if title is not None:
        plt.suptitle(title, y=title_y)
    if save_as is not None:
        plt.margins(0, 0)
        plt.savefig(save_as, dpi=dpi, bbox="tight_inches", transparent=transparent)

    if show_plot:
      plt.show()
    else:
      plt.close(fig)

def plot_sample_grid(nsamples, netG, nz=100, scale=1.0, fixed_noise=None, device="cuda:0", **kwargs):
    if fixed_noise is None:
      noise = Normal(loc=0.0, scale=scale).sample((nsamples, nz, 1, 1))
      noise = noise.to(device)
    else:
        noise = fixed_noise

    with torch.no_grad():
        images = netG(noise).cpu()

    make_image_grid(images=images, **kwargs)

    return noise, images

def apply_sr_image(img_rgb, times=4, lap_sr_dir="lap_sr_models"):
  if times not in [4, 8]:
    print(f"Factor {times} SR does not exist.")
  sr = cv2.dnn_superres.DnnSuperResImpl_create()
  path = f"{lap_sr_dir}/LapSRN_x{times}.pb"
  sr.readModel(path)
  sr.setModel("lapsrn", times)
  img_gbr = img_rgb[::-1]
  #img = cv2.imread("/content/drive/MyDrive/RtBB_experiment_code/test_images/test_img.png")
  upsample_gbr = sr.upsample(img_gbr)
  upsample_rgb = upsample_gbr[::-1]
  return upsample_rgb

def make_gif(im_batch, save_dir, save_name, reflect=False):
    temp_dir = os.path.join(save_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    ims = im_batch.cpu()
    for idx in range(len(ims)):
        im_idx = ims[idx]
        save_image(im_idx, f'{temp_dir}/img_{idx}.png')

    images = []
    filenames = os.listdir(temp_dir)

    for filename in filenames:
        images.append(imageio.imread(f"{temp_dir}/{filename}"))

    if reflect:
        for filename in reversed(filenames):
            images.append(imageio.imread(f"{temp_dir}/{filename}"))

    full_save_path = f"{save_dir}/{save_name}"
    if not ".gif" in save_name:
        full_save_path += ".gif"

    shutil.rmtree(temp_dir)

    print(f"Saving GIF at {full_save_path}")
    imageio.mimsave(full_save_path, images)

def convert_tensor_zero_one_to_numpy_255(tensor_img):
    """
    Convert a Float tensor (0-1 range) image [C, H, W] to a Numpy unit8 format (0-255 range) image [H, W, C].
    If there is a batch dimension, this will remain.
    """
    np_img_array = np.uint8(tensor_img.mul(255.).cpu().numpy()) 
    
    if tensor_img.dim() == 4:
        # [B, C, H, W] -> [B, H, W, C]
        np_img_array = np_img_array.transpose(0, 2, 3, 1)
    else:
        # [C, H, W] -> [H, W, C]
        np_img_array = np_img_array.transpose(1, 2, 0)
    
    return np_img_array

def convert_numpy_255_to_tensor_zero_one(np_img_array):
    """
    Convert a Numpy unit8 format (0-255 range) image [H, W, C]  to a Float tensor (0-1 range) image [C, H, W].
    If there is a batch dimension, this will remain.
    """
    tensor_img = torch.Tensor(np_img_array).float().div(255.)
    if tensor_img.dim() == 4:
        # [B, H, W, C] -> [B, C, H, W]
        tensor_img = tensor_img.permute(0, 3, 1, 2)
    else:
        # [C, H, W] -> [H, W, C]
        tensor_img = tensor_img.permute(2, 0, 1)

    return tensor_img

def right_rotate(lists, num):
    output_list = []
 
    # Will add values from n to the new list
    for item in range(len(lists) - num, len(lists)):
        output_list.append(lists[item])
 
    # Will add the values before
    # n to the end of new list
    for item in range(0, len(lists) - num):
        output_list.append(lists[item])
 
    return output_list

def make_video_mp4(images, video_path, multiply_frames=1):
    # torch tensor to np array
    if torch.is_tensor(images):
        images = convert_tensor_zero_one_to_numpy_255(images)
    # np batch array to list of arrays
    if type(images).__module__ == np.__name__:
        images = [images[i] for i in range(len(images))]
    
    video = imageio.get_writer(video_path, fps=60)
    for img in images:
        # this is needed sometimes not to get glitchy videos
        for _ in range(multiply_frames):
            video.append_data(img)
    
    video.close()
    print("Done making video, saved at:", video_path)