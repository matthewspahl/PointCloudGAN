'''

Matthew Spahl
CS 674 Final Project
May 2022

References:
https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead

@article{achlioptas2017latent_pc,
  title={Learning Representations and Generative Models For 3D Point Clouds},
  author={Achlioptas, Panos and Diamanti, Olga and Mitliagkas, Ioannis and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1707.02392},
  year={2017}
}

'''

import sys
import os
import os.path as osp
import shutil
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
                                        load_all_point_clouds_under_folder

from src.general_utils import plot_3d_point_cloud
from new_generator_discriminator import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='model/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'model_epoch_{}.pth'.format(state['epoch']))
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth'))

# from latent_3d_points
#----------------------------------------------------------------
def generator_noise_distribution(n_samples, ndims, mu, sigma):
    return np.random.normal(mu, sigma, (n_samples, ndims))
#---------------------------------------------------------------

def train(train_data, epochs, batch_size, num_points, G, D, G_optimizer, D_optimizer, args, log_file, log_file2, epoch, generator_updates, gradient_penalty_coefficient):
    """
    Train for one epoch
    """

    G.train()
    D.train()
    batch_index = 0

    total_loss = 0
    epoch_loss_d = 0.
    epoch_loss_g = 0.

    n_examples = train_data.num_examples
    batch_size = batch_size
    n_batches = int(n_examples / batch_size)

    for iteration in range(n_batches):

        batch_array, labels, blank = train_data.next_batch(batch_size)

        batch = torch.from_numpy(batch_array)
        # torch.Tensor

        # uniform point clouds that are from the data, with real SDFs
        batch = batch.to(device)

        real_predictions = []
        fake_predictions = []

        for update in range(generator_updates):

            # minimize negative of mean of discriminator on fake logits
            G_optimizer.zero_grad()
            z = generator_noise_distribution(args.train_batch, args.noise_size, mu=0, sigma=0.2)
            z_float = z.astype(np.float32)
            z_tensor = torch.from_numpy(z_float)
            z_tensor = z_tensor.to(device)

            generator_out = G(z_tensor)
            fake_logits = D(generator_out, None)
            G_loss = -torch.mean(fake_logits)
            # print("g_loss: ", G_loss)
            G_loss.backward()
            G_optimizer.step()

            total_loss += G_loss.abs().item()
            epoch_loss_g += G_loss.abs().item()

            torch.cuda.empty_cache()

        D_optimizer.zero_grad()

        # generates random numbers of size uniform.size(0), latent_size
        # this creates random input for the generator
        z = generator_noise_distribution(args.train_batch, args.noise_size, mu=0, sigma=0.2)
        z_float = z.astype(np.float32)
        # numpy.float32

        z_tensor = torch.from_numpy(z_float)
        z_tensor = z_tensor.to(device)

        # pos: [batch_size, num_points, 3]
        # z: [batch_size, latent_channels]
        generator_out = G(z_tensor)

        # real numbers, one attribute of W-GAN

        # logits are raw output of neurons, before sigmoid
        real_logits = D(batch, None)
        real_predictions.append(torch.mean(real_logits).item())
        # print("real logits: ", real_logits)

        fake_logits = D(generator_out, None)
        fake_predictions.append(torch.mean(fake_logits).item())

        # g_loss = -torch.mean(fake_logits)

        # print("fake logits: ", fake_logits)
        # print("fake probabiliites: ", fake_probabilities)

        # optimizer minimizes
        # discriminator wants to maximize mean(real logits) - mean(fake logits)
        # so we minimize the negative
        D_loss = -(torch.mean(real_logits) - torch.mean(fake_logits))

        # print("mean of fake logits: ", torch.mean(fake_logits))
        # print("mean of real logits: ", torch.mean(real_logits))

        # from towards data science
        # https://towardsdatascience.com/demystified-wasserstein-gan-with-gradient-penalty-ba5e9b905ead
        # ----------------------------------------------------------------------------------
        alpha = torch.rand((args.train_batch, 1, 1), device=device)
        interpolated = alpha * batch + (1 - alpha) * generator_out
        interpolated.requires_grad_(True)
        out_logits = D(interpolated, None)

        # from docs: Computes and returns the sum of gradients of outputs with respect to the inputs.
        grad = torch.autograd.grad(outputs=out_logits, inputs=interpolated,
                                   grad_outputs=torch.ones_like(out_logits),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        # print("grad shape: ", grad.shape) # [batch size, 2048, 3]
        view = grad.contiguous().view(grad.size(0), -1)  # [batch size, 2048 * 3]
        # print("view shape: ", view.shape)

        grad_norm = view.norm(2, dim=1)
        gp = gradient_penalty_coefficient * (grad_norm - 1).pow(2).mean()

        #----------------------------------------------------------------------------------------------
        # print("gp: ", gp)

        loss = D_loss + gp

        # print("d_loss, with penalty: ", loss)
        loss.backward()
        D_optimizer.step()

        total_loss += D_loss.abs().item()
        epoch_loss_d += D_loss.abs().item()

        torch.cuda.empty_cache()

        #total_loss += D_loss.abs().item()
        #epoch_loss_d += D_loss.abs().item()
        batch_index += 1

        print('Epoch [%d/%d], Iter [%d/%d], Training loss: %.4f' % (epoch + 1, epochs, iteration + 1, n_batches, total_loss / (iteration + 1)))
        print()
        log_file.write('{:d} {:.4f} {:.4f}\n'.format(epoch, np.mean(fake_predictions), np.mean(real_predictions)))
        log_file.flush()

        torch.cuda.empty_cache()

    log_file2.write('{:d} {:.4f} {:.4f}\n'.format(epoch, epoch_loss_g, epoch_loss_d))
    log_file2.flush()

    return epoch_loss_g, epoch_loss_d, G, D

def generate(number_samples, generator):
    z = generator_noise_distribution(number_samples, args.noise_size, mu=0, sigma=0.2)
    z_float = z.astype(np.float32)
    z_tensor = torch.from_numpy(z_float)
    z_tensor = z_tensor.to(device)
    out = generator(z_tensor)
    out_cpu = out.detach().cpu().numpy()
    print(out_cpu.shape)
    return out_cpu

def main(args):
    print(sys.argv)
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    LOG_FILE_NAME = "plots/rwgan_training_predictions.csv"
    LOG_FILE_NAME2 = "plots/rwgan_training_loss.csv"
    if args.should_continue:
        log_file1 = open(LOG_FILE_NAME, "a")
        log_file2 = open(LOG_FILE_NAME2, "a")
    else:
        log_file1 = open(LOG_FILE_NAME, "w")
        log_file2 = open(LOG_FILE_NAME2, "a")

    if args.should_continue:
        first_epoch = args.first_epoch
    else:
        first_epoch = 0

    # from original paper
    #-------------------------------------------------------------------------------------------------------------
    # Directory for saving check points and generated data
    top_out_dir = '../data/'

    # Top-dir of where point-clouds are stored.
    top_in_dir = '../data/shape_net_core_uniform_samples_2048/'

    experiment_name = 'raw_gan_with_w_gan_loss'

    n_pc_points = args.num_points  # Number of points per model.
    gradient_penalty_coefficient = 10
    class_name = input('Type the class name (e.g. "chair"): ').lower()

    # Load point-clouds
    syn_id = snc_category_to_synth_id()[class_name]
    class_dir = osp.join(top_in_dir, syn_id)
    all_pc_data = load_all_point_clouds_under_folder(class_dir, n_threads=4, file_ending='.ply', verbose=True)


    print('Shape of DATA =', all_pc_data.point_clouds.shape)
    #----------------------------------------------------------------------------------------------------------------

    G = Point_Cloud_Generator(args.noise_size, [n_pc_points, 3])
    #G = Point_Cloud_Generator_Short(args.noise_size, [n_pc_points, 3])

    #D = MLP_Discriminator_Paper([n_pc_points, 3], 1)
    #D = #PointNet_Discriminator1([n_pc_points, 3], out_channels=1)
    D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], out_channels=1)

    #G_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.0001)
    #D_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.0001)

    G, D = G.to(device), D.to(device)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.9))
    D_optimizer = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.9))

    if args.should_continue:
        generator_checkpoint = torch.load('reverse_checkpoints/generator/model_best.pth')
        discriminator_checkpoint = torch.load('reverse_checkpoints/discriminator/model_best.pth')
        G.load_state_dict(generator_checkpoint['state_dict'])
        G_optimizer.load_state_dict(generator_checkpoint['G_optimizer'])
        D.load_state_dict(discriminator_checkpoint['state_dict'])
        D_optimizer.load_state_dict(discriminator_checkpoint['D_optimizer'])

    generator_updates = args.generator_updates

    print("=> Total params in generator: %.2fM" % (sum(p.numel() for p in G.parameters()) / 1000000.0))
    print("=> Total params in discriminator: %.2fM" % (sum(p.numel() for p in D.parameters()) / 1000000.0))


    # from original
    #--------------------------------------------------------------------------------------------------------
    save_synthetic_samples = True
    save_gan_model = True

    if save_synthetic_samples:
        synthetic_data_out_dir = osp.join(top_out_dir, 'OUT/synthetic_samples/', experiment_name)
        create_dir(synthetic_data_out_dir)

    if save_gan_model:
        train_dir = osp.join(top_out_dir, 'OUT/raw_gan', experiment_name)
        create_dir(train_dir)
    #--------------------------------------------------------------------------------------------------------

    train_stats = []

    for epoch in range(first_epoch, args.epochs):
        torch.cuda.empty_cache()
        all_pc_data = all_pc_data.shuffle_data()
        loss_g, loss_d, G_updated, D_updated = train(all_pc_data, args.epochs, args.train_batch, args.num_points, G, D, G_optimizer, D_optimizer, args, log_file1, log_file2, epoch, generator_updates, gradient_penalty_coefficient)
        G = G_updated
        D = D_updated
        save_checkpoint({"epoch": epoch + 1, "state_dict": G.state_dict(), "G_optimizer": G_optimizer.state_dict()},
                        True, checkpoint_folder=args.g_checkpoint_folder)
        save_checkpoint({"epoch": epoch + 1, "state_dict": D.state_dict(), "D_optimizer": D_optimizer.state_dict()},
                        True, checkpoint_folder=args.d_checkpoint_folder)
        print(f"Epoch{epoch+1:d}. train_generator_loss: {loss_g:.8f}. train_discriminator_loss: {loss_d:.8f}")

        loss = [loss_d, loss_g]
        train_stats.append(loss)

        torch.cuda.empty_cache()

        if epoch % 1 == 0:

# from paper
#--------------------------------------------------------------------------------------------------
            syn_data = generate(args.samples_to_generate, G)
            np.savez(osp.join(synthetic_data_out_dir, 'epoch_' + str(epoch)), syn_data)
            for k in range(args.samples_to_generate):  # plot three (synthetic) random examples.
                plot_3d_point_cloud(syn_data[k][:, 0], syn_data[k][:, 1], syn_data[k][:, 2],
                                    in_u_sphere=True)
#---------------------------------------------------------------------------------------------------

    log_file1.close()
    log_file2.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pointnet++ GAN')
    parser.add_argument('--category', default='chair', type=str, help="The class of model to generate")
    parser.add_argument('--should_continue', default=False, type=str, help="Resume")
    parser.add_argument('--first_epoch', default=1, type=int, help="Epoch for resuming training")
    parser.add_argument("--num_points", default=2048, type=int, help="Number point samples for training")
    parser.add_argument("--generator_updates", default=15, type=int, help="Generator updates per discriminator update")

    # for regular pointnet (mlp discriminator paper), use 50 for batch size
    # for pointnet plus plus, use 10 (more parameters, GTX 1060 6gb cannot fit more)
    parser.add_argument("--train_batch", default=10, type=int, help="Batch size for training")
    parser.add_argument("--noise_size", default=128, type=int, help="Size of generator noise code")
    parser.add_argument("--epochs", default=300, type=int,
                        help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--samples_to_generate", default=3, type=int, help="Samples to generate after each epoch")
    parser.add_argument("--g_checkpoint_folder", default='checkpoints/generator/', type=str, help="Folder for saving generator checkpoints")
    parser.add_argument("--d_checkpoint_folder", default='checkpoints/discriminator/', type=str, help="Folder for saving discriminator checkpoints")
    args = parser.parse_args()
    main(args)