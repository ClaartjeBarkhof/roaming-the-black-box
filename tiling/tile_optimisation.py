import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .tile_objective import TileObjective


class TileOptimiser:
    def __init__(self, netG, lr=0.075, num_steps=600, latent_range=2.0, n_pixels_boundary=7, nz=100,
                 max_weight_pixel=5.0, min_weight_pixel=1.0, max_weight_reg=20.0, min_weight_reg=5.0,
                 latent_std=1.0, initial_noise_factor=0.05, noise_ramp_length=0.75, max_noise=0.05, device="cuda:0"):

        self.device = device

        self.netG = netG

        self.lr = lr
        self.nz = nz
        self.num_steps = num_steps
        self.n_pixels_boundary = n_pixels_boundary

        self.latent_range = latent_range

        self.reg_weight_sched = LinearWeightScheduler(max_weight=max_weight_reg, min_weight=min_weight_reg,
                                                      ramp_frac=0.9,
                                                      num_steps=num_steps, increase_decrease="increase",
                                                      start_end="end")

        self.pixel_weight_sched = LinearWeightScheduler(max_weight=max_weight_pixel, min_weight=min_weight_pixel,
                                                        ramp_frac=0.9,
                                                        num_steps=num_steps, increase_decrease="decrease",
                                                        start_end="start")

        self.noise_scale_sched = NoiseScaleScheduler(num_steps=num_steps, latent_std=latent_std, max_noise=max_noise,
                                                     initial_noise_factor=initial_noise_factor,
                                                     noise_ramp_length=noise_ramp_length)

    @staticmethod
    def init_stats_dict():
        stats = {
            "loss": [],
            "batch_loss": [],

            "batch_pixel_loss": [],
            "batch_weighted_pixel_loss": [],

            "batch_latent_reg_loss": [],
            "batch_weighted_latent_reg_loss": [],

            "batch_noise_sum": [],

            "lr": [],

            "noise_scale": [],

            "reg_weight": [],
            "pixel_weight": [],

            "batch_candidate_image": []
        }

        return stats

    def plot_stats(self, stats):
        num_stats = len(list(stats.keys())) - 1
        ncols = 4
        nrows = int(np.ceil(num_stats / ncols))
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 4.0, nrows * 3.0))

        for i, (stat_name, stats) in enumerate(stats.items()):
            if "image" in stat_name:
                # TODO: add some optimisation plots?
                continue

            r = i // ncols
            c = i % ncols

            if "batch" in stat_name:
                # [B, Num_steps]
                stats_stacked = torch.stack(stats, dim=1)
                # print("stats_stacked shape", stats_stacked.shape)
                # print(stat_name, "len(stats), stats[0].shape", len(stats), stats[0].shape)
                n_idxs = len(stats[-1])
                for idx in range(n_idxs):
                    axs[r, c].plot(np.arange(self.num_steps), stats_stacked[idx].tolist(), label=str(idx))
                axs[r, c].set_title(stat_name)

            else:
                axs[r, c].plot(np.arange(self.num_steps), stats)
                axs[r, c].set_title(stat_name)

        plt.tight_layout()
        plt.show()

    def optimise(self, constraint_imgs, constraint_sides, opposite_transform=False, reduce_black_and_white=False,
                 n_pixels_boundary=None, plot_losses=True, log_every=100):
        if n_pixels_boundary is None:
            n_pixels_boundary = self.n_pixels_boundary

        # -----------------------------------------
        # Init the objective & optimiser etc
        objective = TileObjective(images_to_match=constraint_imgs, latent_range=self.latent_range,
                                  matching_sides=constraint_sides, netG=self.netG,
                                  n_pixels_boundary=n_pixels_boundary, device=self.device,
                                  reduce_black_and_white=reduce_black_and_white,
                                  opposite_transform=opposite_transform)
        random_latents = torch.randn((1, self.nz, 1, 1), device=self.device, dtype=torch.float32, requires_grad=True)

        opt = torch.optim.Adam([random_latents], betas=(0.9, 0.999), lr=self.lr)
        # lr_scheduler = ReduceLROnPlateau(opt, 'min', patience=10, factor=0.5, threshold=0.5)

        stats = self.init_stats_dict()

        # -----------------------------------------
        # Optimise adjacent tile fit
        for step in range(self.num_steps):
            noise_scale = self.noise_scale_sched(step=step)
            reg_weight = self.reg_weight_sched(step=step)
            pixel_weight = self.pixel_weight_sched(step=step)

            # Compute objective
            loss_dict = objective(candidate_latent_batch=random_latents, noise_scale=noise_scale, reg_weight=reg_weight,
                                  pixel_weight=pixel_weight)

            # Step
            opt.zero_grad()
            loss_dict["loss"].backward()
            opt.step()

            # LR scheduler
            lr = opt.param_groups[0]["lr"]
            stats["lr"].append(lr)

            for k, v in loss_dict.items():
                if k == "loss":
                    stats["loss"].append(v.item())
                elif "batch_" in k and not "image":
                    stats[k].append(np.array(v.tolist()))
                else:
                    stats[k].append(v)

            stats["noise_scale"].append(noise_scale)
            stats["reg_weight"].append(reg_weight)
            stats["pixel_weight"].append(pixel_weight)

            if step % log_every == 0 or step + 1 == self.num_steps:
                mes = f'step {step + 1:>4d}/{self.num_steps}: loss {float(loss_dict["loss"].item()):.8f} lr: {lr:.10f}'
                for batch_idx in range(1):
                    batch_idx_total_loss = stats["batch_loss"][-1][batch_idx]
                    mes += f"| {batch_idx}: {batch_idx_total_loss:.2f}"
                print(mes)

        if plot_losses:
            self.plot_stats(stats)

        return stats["batch_candidate_image"]


class NoiseScaleScheduler:
    def __init__(self, num_steps=2000, latent_std=1.0, initial_noise_factor=0.05, noise_ramp_length=0.75,
                 max_noise=0.01):
        self.num_steps = num_steps
        self.latent_std = latent_std
        self.initial_noise_factor = initial_noise_factor
        self.noise_ramp_length = noise_ramp_length
        self.max_noise = max_noise

    def __call__(self, step):
        optim_frac = step / self.num_steps
        noise_scale = self.max_noise * self.latent_std * self.initial_noise_factor * max(0.0,
                                                                                         1.0 - optim_frac / self.noise_ramp_length) ** 2
        return noise_scale


class LinearWeightScheduler:
    def __init__(self, max_weight=10.0, min_weight=0.0, ramp_frac=0.3, num_steps=200, increase_decrease="decrease",
                 start_end="start"):

        self.max_weight = max_weight
        self.min_weight = min_weight

        self.ramp_frac = ramp_frac

        self.num_steps = num_steps

        self.ramp_steps = self.num_steps * self.ramp_frac
        self.non_ramp_steps = self.num_steps - self.ramp_steps

        self.increase_decrease = increase_decrease
        self.start_end = start_end

    def __call__(self, step):
        delta_weight = self.max_weight - self.min_weight
        frac_steps = step / self.num_steps

        # print(frac_steps)

        if self.start_end == "start" and step <= self.ramp_steps:
            frac_ramp = frac_steps / self.ramp_frac

            if self.increase_decrease == "decrease":
                weight = self.max_weight - frac_ramp * delta_weight
            else:
                weight = self.min_weight + frac_ramp * delta_weight

        elif self.start_end == "end" and step >= self.non_ramp_steps:
            ramp_steps = step - self.non_ramp_steps
            frac_ramp = ramp_steps / self.ramp_steps

            if self.increase_decrease == "decrease":
                weight = self.max_weight - frac_ramp * delta_weight
            else:
                weight = self.min_weight + frac_ramp * delta_weight

        elif self.start_end == "start" and frac_steps > self.ramp_frac:
            if self.increase_decrease == "decrease":
                weight = self.min_weight
            else:
                weight = self.max_weight

        else:
            if self.increase_decrease == "decrease":
                weight = self.max_weight
            else:
                weight = self.min_weight

        return weight



