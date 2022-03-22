import os
import numpy as np
import torch
from PIL import Image

import matplotlib.pyplot as plt
from itertools import chain, combinations


def make_target_images(images_to_match, matching_sides):
    """
    Implements the flipping of adjacent tiles that become targets for the new tile to optimise towards.

    Args:
        - images_to_match (List [Tensor])
            List of tensors containing the adjacent tile images [C, H, W].
        - matching_sides (List):
            List of strings containing any combination of the elements 'left', 'right', 'bottom', 'top'.

    Returns:
        - target_images (Dict [Tensor])
            Dictionary with the target images with the correct flips.
    """
    target_images = dict()

    for side, img in zip(matching_sides, images_to_match):
        assert img.dim() == 3, "images must be 3D [C, H, W]"
        if side in ["left", "right"]:
            # Flip left right (dim = W = 2)
            target_images[side] = torch.flip(img, dims=(2,))
        else:
            # Flip top bottom (dim = H = 1)
            target_images[side] = torch.flip(img, dims=(1,))

    return target_images


def make_target_loss_masks(matching_sides, n_pixels_boundary, H=64, W=64):
    """
    This function creates a mask for each adjacent tile constraint, while taking into account other constraints as well.
    The mask will be used in the objective function to mask the flipped adjacent tile which is used for comparison.

    Args:
        - matching_sides (List):
            List of strings containing any combination of the elements 'left', 'right', 'bottom', 'top'.
        - n_pixels_boundary (int):
            Number of pixels to use in the boundary condition.
    """

    empty_loss_mask = torch.zeros(1, H, W)

    right_loss_mask = empty_loss_mask.clone()
    right_loss_mask[:, :, -n_pixels_boundary:] = 1.0

    left_loss_mask = empty_loss_mask.clone()
    left_loss_mask[:, :, :n_pixels_boundary] = 1.0

    bottom_loss_mask = empty_loss_mask.clone()
    bottom_loss_mask[:, -n_pixels_boundary:, :] = 1.0

    top_loss_mask = empty_loss_mask.clone()
    top_loss_mask[:, :n_pixels_boundary, :] = 1.0

    left_lower_triangle = torch.tril(torch.ones_like(top_loss_mask))
    right_lower_triangle = left_lower_triangle.flip(dims=(2,))
    left_upper_triangle = left_lower_triangle.flip(dims=(1,))
    right_upper_triangle = right_lower_triangle.flip(dims=(1,))

    loss_masks = dict()

    # 1: Top - right
    if set(matching_sides) == {"top", "right"}:
        top = top_loss_mask.logical_and(left_upper_triangle)
        right = right_loss_mask.logical_and(right_lower_triangle)

        loss_masks["top"] = top
        loss_masks["right"] = right

        # 2: Top - left
    elif set(matching_sides) == {"top", "left"}:
        top = top_loss_mask.logical_and(right_upper_triangle)
        left = left_loss_mask.logical_and(left_lower_triangle)

        loss_masks["top"] = top
        loss_masks["left"] = left

    # 3: Top - left - right
    elif set(matching_sides) == {"top", "left", "right"}:
        top = top_loss_mask.logical_and(right_upper_triangle).logical_and(left_upper_triangle)
        left = left_loss_mask.logical_and(left_lower_triangle)
        right = right_loss_mask.logical_and(right_lower_triangle)

        loss_masks["top"] = top
        loss_masks["right"] = right
        loss_masks["left"] = left

    # 4: Top - left - bottom
    elif set(matching_sides) == {"top", "left", "bottom"}:

        top = top_loss_mask.logical_and(right_upper_triangle)
        left = left_loss_mask.logical_and(left_lower_triangle).logical_and(left_upper_triangle)
        bottom = bottom_loss_mask.logical_and(right_lower_triangle)

        loss_masks["top"] = top
        loss_masks["left"] = left
        loss_masks["bottom"] = bottom

    # 5: Top - right - bottom
    elif set(matching_sides) == {"top", "right", "bottom"}:
        top = top_loss_mask.logical_and(left_upper_triangle)
        right = right_loss_mask.logical_and(right_lower_triangle).logical_and(right_upper_triangle)
        bottom = bottom_loss_mask.logical_and(left_lower_triangle)

        loss_masks["top"] = top
        loss_masks["right"] = right
        loss_masks["bottom"] = bottom

    # 6: Top - right - bottom - left
    elif set(matching_sides) == {"top", "right", "bottom", "left"}:
        top = top_loss_mask.logical_and(left_upper_triangle).logical_and(right_upper_triangle)
        right = right_loss_mask.logical_and(right_lower_triangle).logical_and(right_upper_triangle)
        bottom = bottom_loss_mask.logical_and(left_lower_triangle).logical_and(right_lower_triangle)
        left = left_loss_mask.logical_and(left_upper_triangle).logical_and(left_lower_triangle)

        loss_masks["top"] = top
        loss_masks["right"] = right
        loss_masks["left"] = left
        loss_masks["bottom"] = bottom

    # 7: Right - bottom
    elif set(matching_sides) == {"right", "bottom"}:
        right = right_loss_mask.logical_and(right_upper_triangle)
        bottom = bottom_loss_mask.logical_and(left_lower_triangle)

        loss_masks["right"] = right
        loss_masks["bottom"] = bottom

    # 8: Left - bottom
    elif set(matching_sides) == {"left", "bottom"}:
        left = left_loss_mask.logical_and(left_upper_triangle)
        bottom = bottom_loss_mask.logical_and(right_lower_triangle)

        loss_masks["left"] = left
        loss_masks["bottom"] = bottom

    # 9: Right - left - bottom
    elif set(matching_sides) == {"right", "left", "bottom"}:
        left = left_loss_mask.logical_and(left_upper_triangle)
        bottom = bottom_loss_mask.logical_and(right_lower_triangle).logical_and(left_lower_triangle)
        right = right_loss_mask.logical_and(right_upper_triangle)

        loss_masks["right"] = right
        loss_masks["left"] = left
        loss_masks["bottom"] = bottom

    # 10: Right
    elif set(matching_sides) == {"right"}:
        right = right_loss_mask

        loss_masks["right"] = right

    # 11: Left
    elif set(matching_sides) == {"left"}:
        left = left_loss_mask

        loss_masks["left"] = left

    # 12: Top
    elif set(matching_sides) == {"top"}:
        top = top_loss_mask

        loss_masks["top"] = top

    # 13: Bottom
    elif set(matching_sides) == {"bottom"}:
        bottom = bottom_loss_mask

        loss_masks["bottom"] = bottom

    # 14: Top - bottom
    elif set(matching_sides) == {"top", "bottom"}:
        bottom = bottom_loss_mask
        top = top_loss_mask

        loss_masks["top"] = top
        loss_masks["bottom"] = bottom

    # 15: Top - bottom
    elif set(matching_sides) == {"left", "right"}:
        left = left_loss_mask
        right = right_loss_mask

        loss_masks["right"] = right
        loss_masks["left"] = left

    else:
        raise ValueError("Sides are not correct, need to be a (sub)set of ['left', 'right', 'top', 'bottom']")

    return loss_masks


def make_target_images_and_loss_masks(images_to_match, matching_sides, n_pixels_boundary, plot=False):
    """
    Implements calling for target images (adjacent tiles flipped in the correct way) and for mask creation
    that can be used together to construct a loss for multiple constraints at a time.

    Args:
        - images_to_match (List [Tensor])
            List of tensors containing the adjacent tile images [C, H, W].
        - matching_sides (List):
            List of strings containing any combination of the elements 'left', 'right', 'bottom', 'top'.

    Returns:
        - target_images_list (List [Tensor]):
            List with the target images with the correct flips.
        - loss_masks_list: (List [Tensor] [1, H, W]):
            List with tensor loss masks. Order matches with target_images_list.
    """

    # images_to_match: images in adjacent tiles
    # on which side these tiles live

    C, H, W = images_to_match[0].shape
    loss_masks = make_target_loss_masks(matching_sides=matching_sides, n_pixels_boundary=n_pixels_boundary, H=H, W=W)
    target_images = make_target_images(images_to_match, matching_sides)

    if plot:
        for k, v in loss_masks.items():
            plt.imshow(v[0])
            plt.title(k)
        for k, v in target_images.items():
            plt.imshow(v)
            plt.title(k)

    loss_masks_list = []
    target_images_list = []

    # Make a list with same order
    for side in ['left', 'right', 'top', 'bottom']:
        if side in loss_masks:
            loss_masks_list.append(loss_masks[side])
            target_images_list.append(target_images[side])

    return target_images_list, loss_masks_list


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def export_all_masks_as_png(save_dir, n_pixels_boundary=8, H=64, W=64, plot=False):
    print("Generating all loss masks and saving to", save_dir)

    all_sides = ["top", "right", "bottom", "left"]
    all_loss_masks = dict()

    for s in powerset(all_sides):
        if len(s) > 0:
            name = "_".join(list(s))
            loss_masks = make_target_loss_masks(matching_sides=list(s), n_pixels_boundary=n_pixels_boundary, H=H, W=W)
            all_loss_masks[name] = loss_masks

    if save_dir[-1] == "/":
        save_dir = save_dir[:-1]

    os.makedirs(save_dir, exist_ok=True)

    for objective, image_dict in all_loss_masks.items():
        for side, target_mask_img in image_dict.items():
            os.makedirs(f"{save_dir}/{objective}", exist_ok=True)
            save_name = f"{save_dir}/{objective}/{side}.png"

            # [1, H, W] -> [H, W] (discard channel info, saving as mode = L)
            save_img_arr = (target_mask_img[0].cpu().numpy() * 255.).astype(np.uint8)

            im = Image.fromarray(save_img_arr, mode="L")
            im.save(save_name)

            if plot:
                plt.imshow(save_img_arr)
                plt.title(save_name)
                plt.show()


if __name__ == "__main__":
    d = "/Users/claartje/Dropbox/Werk/bakken_baeck_2022/Code/roaming-the-black-box/src/workers/tile/target_masks"

    export_all_masks_as_png(save_dir=d, n_pixels_boundary=8, H=64, W=64, plot=False)