import torch
import torch.nn as nn
import torch.nn.functional as F

from .tile_target_masks import make_target_images_and_loss_masks

class TileObjective(nn.Module):
    """
    Args:
        - images_to_match - List [ List [ Tensor ] ]:
            e.g. [[img0], [img1, img2], ..., [img0, img3]]
            A list of lists, where every sublist contains the constraints for one
            tile optimisation and the outer list thus represents the batch dimension.
        - matching_sides - List [ List [ str ] ]:
            e.g. [["left"], ["left", "top"], ..., ["right", "bottom"]]
            List of lists, matching the lengths of <images_to_match> defining which sides
            (defined as strings) the images need to be matched of the tile that is being optimised
            e.g. 'left' means that there is an image to the 'left' that needs to be matched.
        - netG - nn.Module:
            A generator network that maps noise to images (GAN).
        - n_pixels_boundary - int:
            Number of pixels to use in the boundary condition.
        - latent_range - int:
            maximum values the latent representation can take on (used to clamp)
        - device - str:
            which device to use, e.g. 'cpu' or 'cuda:0'
        - opposite_transform - bool:
            whether to invert the target image to optimise towards
        - reduce_black_and_white - bool
            whehter to compare the new tile and constraining tiles in a black and white projection (matching brightness)
        - plot_objective - bool
            whether to plot the target images and loss masks for debug
    """
    def __init__(self, images_to_match, matching_sides, netG, n_pixels_boundary=10, latent_range=1.5,
                 device="cuda:0", opposite_transform=False, reduce_black_and_white=False, plot_objective=False):
        super(TileObjective, self).__init__()

        # Assertions
        assert len(images_to_match) == len(matching_sides), \
            "length of images_to_match should equal the length of matching_sides"
        assert type(images_to_match[0]) == type(matching_sides[0]) == list, \
            "images_to_match and matching_sides should be lists of lists"

        for sublist in matching_sides:
            for ms in sublist:
                assert ms in ["left", "right", "top", "bottom"], \
                    "matching side needs to be one of: [left, right, top, bottom]"

        for sublist in images_to_match:
            for im in sublist:
                assert im.dim() == 3, "individual images must be 3D"

        # Craft targets and loss masks
        targets, loss_masks = [], []
        for batch_idx, (ms, im) in enumerate(zip(matching_sides, images_to_match)):
            t, m = make_target_images_and_loss_masks(images_to_match=im, matching_sides=ms,
                                                     n_pixels_boundary=n_pixels_boundary, plot=plot_objective)
            targets.append([t_.to(device) for t_ in t])
            loss_masks.append([m_.to(device) for m_ in m])

        self.targets = targets
        self.loss_masks = loss_masks

        print("len(targets) [B] len(targets[0]) [N_constraints_0]", len(targets), len(targets[0]))
        print("len(loss_masks) [B] len(loss_masks[0]) [N_constraints_0]", len(loss_masks), len(loss_masks[0]))

        # TODO: this might vary for different elements in the batch...
        self.reduce_black_and_white = reduce_black_and_white
        self.opposite_transform = opposite_transform
        self.latent_range = latent_range

        self.batch_size = len(self.loss_masks)

        # Set the generator in the correct mode, make sure gradients are off!
        self.netG = netG
        self.netG.eval()
        for param in self.netG.parameters():
            param.requires_grad = False

    @staticmethod
    def rgb_to_bw(image_batch):
        # 0.299 ∙ Red + 0.587 ∙ Green + 0.114 ∙ Blue
        return 0.299 * image_batch[:, 0, :, :] + 0.587 * image_batch[:, 1, :, :] + 0.114 * image_batch[:, 2, :, :]

    def black_and_white_l1(self, candidate_batch, target_batch):
        # greyscale = 0.2125 * R + 0.7154 * G + 0.0721 * B
        candidate_bw = self.rgb_to_bw(candidate_batch)
        target_bw = self.rgb_to_bw(target_batch)
        # make channel dim = 1
        l1_loss = F.l1_loss(candidate_bw, target_bw, reduction="none").unsqueeze(1)
        return l1_loss

    def forward(self, candidate_latent_batch, noise_scale, reg_weight, pixel_weight):
        """
        Compute the loss as:
            Z_noisy = clamped(Z + scaled_noise) to max latent range
            X_noisy = netG(Z_noisy)
            Loss = pixel_weight * image_loss(X_noisy, adjacent_tile_constraints) + reg_weight * L1(Z_noisy)

        Args:
            candidate_latent_batch - Tensor [B, NZ]:
                A batch latent to be optimised. Each element in the batch belongs to a separate tile with its
                own constraints. Each element in the batch is matched with an element in the
                loss_masks and targets lists.
            noise_scale - int:
                Scale for added noise.
            reg_weight - int:
                Weight for the latent regularisation loss (how far the latent is off the center of standard Normal).
            pixel_weight - int:
                Weight for the pixel loss (boundary objective).

        Returns:
            loss_dict - Dict [ Tensor ]
                A dictionary containing losses. The 'loss' key yields the loss to perform backprop with.
        """
        # Add noise, with noise_scale weight
        noise = torch.randn_like(candidate_latent_batch) * noise_scale
        batch_noise_sum = torch.abs(noise).mean(dim=(1, 2, 3))
        candidate_latent_batch = candidate_latent_batch + noise
        candidate_latent_batch = torch.clamp(candidate_latent_batch, min=-self.latent_range, max=self.latent_range)

        # Regularisation loss is the sum of the absolute values of the latents
        # which is equivalent to the L1 distance to the mean of a Standard Normal
        batch_latent_reg_loss = torch.abs(candidate_latent_batch).mean(dim=(1, 2, 3))
        batch_weighted_latent_reg_loss = reg_weight * batch_latent_reg_loss

        # Forward pass
        batch_candidate_image = self.netG(candidate_latent_batch)

        batch_image_loss = []

        # Compare the pixel values with L1 distance (absolute distance) and mask for the boundary
        # we loop over lists of constraints (one sublist contains the constraints for one element in the batch), which
        # might be more than 1!
        for batch_idx, (targets, loss_masks) in enumerate(zip(self.targets, self.loss_masks)):

            image_loss = None  # just to initialise the addition cycle

            # There might be multiple targets for one image (multiple constraining sides)

            # t [3, H, W]
            # m [1, H, W]
            # batch_candidate_image[batch_idx].unsqueeze(0) [1, 3, H, W]
            # image_loss_i [1, C, H, W], C might be 1 or 3 depending on reduce_black_and_white
            for t, m in zip(targets, loss_masks):
                # reduce_black_and_white & opposite_transform are NOT mutually exclusive
                if self.reduce_black_and_white:
                    if self.opposite_transform:
                        image_loss_i = self.black_and_white_l1(batch_candidate_image[batch_idx].unsqueeze(0),
                                                               1.0 - t.unsqueeze(0))
                    else:
                        image_loss_i = self.black_and_white_l1(batch_candidate_image[batch_idx].unsqueeze(0),
                                                               t.unsqueeze(0))
                elif self.opposite_transform:
                    image_loss_i = F.l1_loss(batch_candidate_image[batch_idx].unsqueeze(0), 1.0 - t.unsqueeze(0),
                                             reduction="none")
                else:
                    image_loss_i = F.l1_loss(batch_candidate_image[batch_idx].unsqueeze(0), t.unsqueeze(0),
                                             reduction="none")

                # Init or add to existing
                if image_loss is None:
                    image_loss = image_loss_i * m
                else:
                    image_loss = image_loss + image_loss_i * m

            batch_image_loss.append(image_loss)

        # [B, C, W, H]
        batch_image_loss = torch.cat(batch_image_loss)

        # Mean [B, C, W, H] -> [B]
        batch_pixel_loss = batch_image_loss.mean(dim=(1, 2, 3))
        batch_weighted_pixel_loss = batch_pixel_loss * pixel_weight

        # Loss
        batch_loss = batch_weighted_pixel_loss + batch_weighted_latent_reg_loss
        reduced_loss = batch_loss.mean()

        loss_dict = {
            "loss": reduced_loss,
            "batch_loss": batch_loss,

            "batch_pixel_loss": batch_pixel_loss,  # pixel loss
            "batch_weighted_pixel_loss": batch_weighted_pixel_loss,  # scaled pixel loss

            "batch_latent_reg_loss": batch_latent_reg_loss,  # reg
            "batch_weighted_latent_reg_loss": batch_weighted_latent_reg_loss,  # scaled reg

            "batch_noise_sum": batch_noise_sum,  # size of the noise

            "batch_candidate_image": batch_candidate_image
        }

        # Detach and copy to cpu all but 'loss' key which is used for backprop
        loss_dict_cpu = dict()
        for k, v in loss_dict.items():
            if k != "loss":
                loss_dict_cpu[k] = v.detach().cpu()
            else:
                loss_dict_cpu[k] = v

        return loss_dict_cpu