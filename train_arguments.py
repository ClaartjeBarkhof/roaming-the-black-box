import configargparse


def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')


def prepare_parser(raw_args=None):
    parser = configargparse.ArgParser(description='toy Beta-VAE')

    parser.add_argument('--config_file', required=False, is_config_file=True, help='config file path')

    parser.add_argument('--checkpointing', default=True, type=str2bool, help='checkpointing or not True/False')

    parser.add_argument('--device', default="cpu", type=str, help='which device to use (cpu or cuda:0)')

    parser.add_argument('--adam_beta', default=0.5, type=float, help='Beta 1 for Adam optimiser')
    parser.add_argument('--lr', default=0.0002, type=float, help='Beta 1 for Adam optimiser')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of train epochs')

    parser.add_argument('--ndf', default=64, type=int, help='Number of Discriminator filters')
    parser.add_argument('--ngf', default=64, type=int, help='Number of Generator filters')
    parser.add_argument('--nz', default=100, type=int, help='Number of latent dimensions')

    parser.add_argument('--nchannels', default=3, type=int, help='Number of channels in image data')
    parser.add_argument('--image_dim', default=64, type=int, help="Image dim (H, W)")

    parser.add_argument('--batch_size', default=128, type=int, help="Train batch size")

    parser.add_argument('--data_dir', default="/content/drive/MyDrive/RtBB_experiment_code/Datasets/tile_dataset",
                        type=int, help="Image dim (H, W)")
    parser.add_argument('--nworkers', default=2, type=int, help="Number of workers for data loading")

    parser.add_argument('--checkpoint_save_name', default="tile-gan.pt", type=str, help='Where to save checkpoints.')
    parser.add_argument('--output_dir', default="/content/drive/MyDrive/RtBB_experiment_code/tile-GAN-output", type=str, help="Default output dir.")

    parser.add_argument('--continue_train', default=False, type=str2bool, help='Continue training from a checkpoint')
    parser.add_argument('--continue_train_path', default="", type=str, help='Path to checkpoint to continue training from')

    args = parser.parse_args(raw_args)

    for k, v in vars(args).items():
        print(k, ":", v)

    return args
