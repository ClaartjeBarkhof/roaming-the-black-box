import os

import torch
from torch.distributions import Normal
from torchvision.utils import save_image

import numpy as np
import matplotlib.pyplot as plt

import imageio


def make_image_grid(images, ncols, cell_size=1.5, un_normalize=False, set_idx_title=False, border=False, wspace=0.0,
                    hspace=0.0, title=None, save_as=None):
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
        plt.suptitle(title)
    if save_as is not None:
        plt.savefig(save_as, dpi=300, bbox="tight_inches")


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