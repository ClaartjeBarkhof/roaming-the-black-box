import matplotlib.pyplot as plt
import random
import torch

from .tile_optimisation import TileOptimiser


class TileGridSystem:
    def __init__(self, netG, ncols=8, nrows=4, nz=100, device="cuda:0", initial_latent=None,
                 lr=0.075, num_steps=600, latent_range=2.0, n_pixels_boundary=7,
                 max_weight_pixel=5.0, min_weight_pixel=1.0, max_weight_reg=20.0, min_weight_reg=5.0,
                 latent_std=1.0, initial_noise_factor=0.05, noise_ramp_length=0.75, max_noise=0.05):

        self.netG = netG
        self.nz = nz
        self.device = device

        self.ncols = ncols
        self.nrows = nrows
        self.nimgs = ncols * nrows

        self.initial_latent = initial_latent

        self.tile_order = []
        self.imgs = dict()
        self.optimisations = dict()

        self.tile_optimiser = TileOptimiser(lr=lr,
                                            nz=nz,
                                            num_steps=num_steps,
                                            latent_range=latent_range,
                                            n_pixels_boundary=n_pixels_boundary,
                                            max_weight_pixel=max_weight_pixel,
                                            min_weight_pixel=min_weight_pixel,
                                            max_weight_reg=max_weight_reg,
                                            min_weight_reg=min_weight_reg,
                                            latent_std=latent_std,
                                            initial_noise_factor=initial_noise_factor,
                                            noise_ramp_length=noise_ramp_length,
                                            max_noise=max_noise,
                                            device=device)

    def idx_to_row_col(self, idx):
        row = idx // self.ncols
        col = idx % self.ncols
        return row, col

    def row_col_to_idx(self, row, col):
        idx = self.ncols * row + col
        return idx

    def get_constraint_imgs_and_sides(self, row, col):
        idx = self.row_col_to_idx(row, col)

        idx_top = idx - self.ncols
        idx_bottom = idx + self.ncols
        idx_left = idx - 1
        idx_right = idx + 1

        sides = ["top", "bottom", "left", "right"]
        side_idxs = [idx_top, idx_bottom, idx_left, idx_right]

        constraint_imgs = []
        constraint_sides = []
        for side_idx, side in zip(side_idxs, sides):

            side_row, side_col = self.idx_to_row_col(side_idx)

            # if the adjacent tile exists in grid and has been filled yet, add as constraint
            if side_idx >= 0 and side_idx < self.nimgs and side_idx in self.imgs and (
                    side_row == row or side_col == col):
                constraint_imgs.append(self.imgs[side_idx])
                constraint_sides.append(side)

        return constraint_imgs, constraint_sides

    def plot_grid(self, tile_size=1.5):
        fig, axs = plt.subplots(ncols=self.ncols, nrows=self.nrows,
                                figsize=(self.ncols * tile_size, self.nrows * tile_size))
        fig.patch.set_facecolor('#2D2D2D')  # dark BB grey

        for idx, img in self.imgs.items():
            r, c = self.idx_to_row_col(idx)
            # permute necessary?
            axs[r, c].imshow(img.cpu().permute(1, 2, 0), vmin=0.0, vmax=1.0, aspect="auto")

        for r in range(self.nrows):
            for c in range(self.ncols):
                axs[r, c].axis("off")

        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.show()

    def sample_random(self, idx=None, latent=None, plot_grid_after=True):
        if idx is None:
            idx = random.randint(0, self.nimgs - 1)

        if latent is None:
            noise = torch.randn((1, self.nz, 1, 1)).to(self.device)
        else:
            if latent.dim() == 3:
                noise = latent.unsqueeze(0)
            else:
                noise = latent

        r_img = self.netG(noise)

        self.tile_order.append(idx)
        self.optimisations[idx] = None
        self.imgs[idx] = r_img[0].detach().cpu()
        print("sample random self.imgs[idx] shape", self.imgs[idx].shape)

        if plot_grid_after:
            self.plot_grid()

        return idx

    def check_full(self):
        if len(list(self.imgs.keys())) == self.nimgs:
            return True
        else:
            return False

    def get_exist_idxs(self):
        return list(self.imgs.keys())

    def get_random_new_empty_idx(self):
        """Get a random new tile idx to create, this can
        be next to an existing tile or not."""
        exist_idxs = self.get_exist_idxs()
        if len(exist_idxs) > 0:
            idx = random.choice(exist_idxs)
        else:
            idx = random.randint(0, self.nimgs - 1)
        return idx

    def get_adjacent_idxs(self, idx):
        r_orig, c_orig = self.idx_to_row_col(idx)
        adjacent_idxs = [idx + 1, idx - 1, idx + self.ncols, idx - self.ncols]
        valid_adjacent_idxs = []

        for a_idx in adjacent_idxs:
            r, c = self.idx_to_row_col(a_idx)
            if a_idx < self.nimgs and a_idx >= 0 and (r == r_orig or c == c_orig):
                valid_adjacent_idxs.append(a_idx)

        return valid_adjacent_idxs

    def get_random_expansion_idx(self):
        exist_idxs = self.get_exist_idxs()

        # Either the grid is full
        if self.check_full():
            print("Grid is full!")
            return_idx = None

        # Or there is exist at least one tile
        elif len(exist_idxs) > 0:
            random_expand_idx = None
            while random_expand_idx is None:
                # get random tile that does exist
                random_exist_idx = random.choice(exist_idxs)  # can also be done in order
                # find its adjacent indices
                expand_idxs = self.get_adjacent_idxs(idx=random_exist_idx)
                # check which of these adjacent index tiles do not exist yet
                non_exist_expand_idx = []
                for idx in expand_idxs:
                    if idx not in exist_idxs:
                        non_exist_expand_idx.append(idx)
                print("non_exist_expand_idx", non_exist_expand_idx)
                # if one is found: return it as a valid expansion idx direction
                if len(non_exist_expand_idx) > 0:
                    random_expand_idx = random.choice(non_exist_expand_idx)
            return_idx = random_expand_idx

        # Or there exists nothing yet
        else:
            return_idx = self.get_random_new_empty_idx()

        return return_idx

    def erase_idx(self, idx):
        if idx in self.imgs:
            del self.imgs[idx]
            del self.optimisations[idx]
            self.tile_order.remove(idx)
        else:
            print(f"idx = {idx} not in grid!")

    def expand(self, idx=None, opposite_transform=False, reduce_black_and_white=False, plot_losses=True, log_every=100,
               plot_grid_after=True):
        if idx is None:
            idx = self.get_random_expansion_idx()
            print(f"Random expansion to idx = {idx}")

        row, col = self.idx_to_row_col(idx)
        constraint_imgs, constraint_sides = self.get_constraint_imgs_and_sides(row, col)

        # If there are no constraints, we can just sample!
        if len(constraint_imgs) == 0:
            print("Sample random, because there exist no constraints")
            idx = self.sample_random(idx=idx, latent=self.initial_latent, plot_grid_after=False)

        # Otherwise optimise towards a new, fitting tile!
        else:
            # constraint_imgs, constraint_sides
            print("constraint_sides", constraint_sides)
            optim_imgs = self.tile_optimiser.optimise(constraint_imgs=[constraint_imgs],
                                                      constraint_sides=[constraint_sides],
                                                      opposite_transform=opposite_transform,
                                                      reduce_black_and_white=reduce_black_and_white,
                                                      plot_losses=plot_losses,
                                                      log_every=log_every)

            self.imgs[idx] = optim_imgs[-1][0].detach().cpu()
            self.optimisations[idx] = [im[0].detach().cpu() for im in optim_imgs]
            self.tile_order.append(idx)

        if plot_grid_after:
            self.plot_grid()

        return idx