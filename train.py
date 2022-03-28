# Adapted from: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from GAN import Generator, Discriminator
from dataset import ImageDataset
from utils import make_image_grid


# Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(train_dataloader, data_dir, lr=0.0002, nz=100, num_epochs=200, beta1=0.5, ngpu=1, ndf=64, ngf=64, nc=3,
          image_dim=64, device="cuda:0", print_model=False, continue_train=False, continue_train_path="",
          save_state_dict_path=None, just_load_checkpoint=False):

    if continue_train:
        checkpoint = torch.load(continue_train_path)
        hp = checkpoint["hp"]
        nz = hp["nz"]
        lr = hp["lr"]
        beta1 = hp["beta1"]
        ndf = hp["ndf"]
        ngf = hp["ngf"]
        nc = hp["nc"]
        image_dim = hp["image_dim"]
    else:
        hp = dict(nz=nz, lr=lr, beta1=beta1, ndf=ndf, ngf=ngf, nc=nc, image_dim=image_dim, data_dir=data_dir)

    if save_state_dict_path is not None:
        if ".pt" in save_state_dict_path:
            save_state_dict_path = save_state_dict_path.replace(".pt", "")
        while os.path.exists(save_state_dict_path):
            save_state_dict_path += "-1"
        print(f"Saving results to: {save_state_dict_path}")
        os.makedirs(save_state_dict_path, exist_ok=True)

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Create the generator
    netG = Generator(ngpu=ngpu, ngf=ngf, nz=nz, image_dim=image_dim, nc=nc).to(device)

    # Create the Discriminator
    netD = Discriminator(ngpu=ngpu, ndf=ndf, nc=nc, image_dim=image_dim).to(device)

    # Print the model
    if print_model:
        print(netD)
        print(netG)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    if not continue_train:
        print("Initialising new weights.")
        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.02.
        netG.apply(weights_init)

        # Apply the weights_init function to randomly initialize all weights
        #  to mean=0, stdev=0.2.
        netD.apply(weights_init)

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        start_epoch = 0

    else:
        print("Loading weights from state dict.")
        checkpoint = torch.load(continue_train_path)
        G_losses = checkpoint["G_losses"]
        D_losses = checkpoint["D_losses"]
        img_list = checkpoint["img_list"]

        iters = checkpoint["iters"]
        start_epoch = checkpoint["epoch"]

        netG.load_state_dict(state_dict=checkpoint["netG_state_dict"])
        netD.load_state_dict(state_dict=checkpoint["netD_state_dict"])
        optimizerG.load_state_dict(state_dict=checkpoint["optimizerG_state_dict"])
        optimizerD.load_state_dict(state_dict=checkpoint["optimizerD_state_dict"])

        if just_load_checkpoint:
            return netG, netD

    save_dict = None

    # Just to be sure
    netG.train()
    for param in netG.parameters():
        param.requires_grad = True

    netD.train()
    for param in netD.parameters():
        param.requires_grad = True

    print("-" * 10, " HP ", "-" * 10)
    for k, v in hp.items():
        print(k, v)
    print("-" * 25)



    if continue_train is None:
        print(f"Resuming training training loop [epoch {start_epoch} / iter {iters}]")
    else:
        print(f"Resuming training training loop [epoch {start_epoch} / iter {iters}]")

    # For each epoch
    for epoch in range(start_epoch, num_epochs + start_epoch):
        # For each batch in the dataloader
        i = 0
        for i, data in enumerate(train_dataloader, 0):

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            ## Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)

            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)

            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(train_dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # print("real min max", real_cpu.min().item(), real_cpu.max().item())
                # print("fake min max", fake.min().item(), fake.max().item())

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            iters += 1

        # -------------------------------------------------------------
        # END OF EPOCH

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=2)
        img_list.append(img)  # , normalize=True

        # Plot
        img = img.permute(1, 2, 0)
        # img = (img + 1.0) / 2.0
        plt.imshow(img, vmin=0.0, vmax=1.0)
        plt.title(f"Epoch {epoch} | iter {i} | total iter {iters} ")
        plt.axis("off")
        plt.show()

        if save_state_dict_path is not None:
            print(f"End of epoch {epoch} - Saving state and info in {save_state_dict_path}")
            save_state_dict_path_current = save_state_dict_path + f"/epoch-{epoch}.pt"
            save_dict = dict(G_losses=G_losses, D_losses=D_losses, hp=hp,
                             img_list=img_list,
                             epoch=epoch, step=i, iters=iters,
                             netG_state_dict=netG.state_dict(),
                             netD_state_dict=netD.state_dict(),
                             optimizerG_state_dict=optimizerG.state_dict(),
                             optimizerD_state_dict=optimizerD.state_dict())
            torch.save(save_dict, save_state_dict_path_current)

    print("Done training! Returning save_dict...")

    return save_dict


def run_train(config, plot_data=True):
    synth_dataset = ImageDataset(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_dim=config.image_dim,
        pin_memory=True if "cuda" in config.device else False,
        num_workers=config.nworkers)

    synth_dataset.setup()
    train_loader = synth_dataset.train_dataloader()

    if plot_data:
        # Plot some training images
        real_batch = next(iter(train_loader))
        make_image_grid(real_batch[0][:64], ncols=8, cell_size=1.0, un_normalize=False, set_idx_title=False,
                        border=False, wspace=0.0, hspace=0.0, save_as=None, title="Train data samples")

    ngpu = 1 if "cuda" in config.device else 0
    save_dict = train(train_loader, config.data_dir, lr=config.lr, nz=config.nz, num_epochs=config.num_epochs,
                      beta1=config.adam_beta,
                      ngpu=ngpu, ndf=config.ndf, ngf=config.ngf, nc=config.nchannels, image_dim=config.image_dim,
                      device=config.device, print_model=False, continue_train=config.continue_train,
                      continue_train_path=config.continue_train_path,
                      save_state_dict_path=f"{config.output_dir}/{config.checkpoint_save_name}",
                      just_load_checkpoint=False)

    return save_dict
