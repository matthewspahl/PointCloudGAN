import sys
import argparse
import os
from src.general_utils import plot_3d_point_cloud
from new_generator_discriminator import *
import matplotlib.pyplot as plt
import csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_loss(filename, minibatches, updates):
    """
    Plot the generator and discriminator loss per epoch
    """

    data = []
    epochs = []
    g_loss = []
    d_loss = []

    #source: https://docs.python.org/3/library/csv.html
    with open(filename, newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter= ' ')

        for line in dataReader:
            data.append(line)

    # convert strings to floats
    for row in range(len(data)):
        for column in range(3):
            data[row][column] = float(data[row][column])

    for row in range(len(data)):
        epochs.append(data[row][0])
        g_loss.append(data[row][1]/minibatches)
        d_loss.append(data[row][2]/minibatches)

    #plt.plot(epochs, g_loss, label='Generator loss')
    #plt.ylabel('Loss')
    #plt.xlabel('Epoch')
    #plt.plot(epochs, d_loss, label='Discriminator loss')
    #plt.legend()
    #plt.show()

    # loss per epoch
    plt.plot(epochs, g_loss, label='Generator loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(epochs, d_loss, label='Discriminator loss')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()

    # loss per update
    for value in range(len(d_loss)):
        d_loss[value] = d_loss[value] / updates

    plt.plot(epochs, g_loss, label='Generator loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.plot(epochs, d_loss, label='Discriminator loss')
    plt.ylim(0, 10)
    plt.legend()
    plt.show()

def plot_predictions(filename):
    """
    Plot the discriminator predictions on real and generated data
    """

    data = []
    batches = []
    fake = []
    real = []

    #source: https://docs.python.org/3/library/csv.html
    with open(filename, newline='') as csvfile:
        dataReader = csv.reader(csvfile, delimiter= ' ')

        for line in dataReader:
            data.append(line)

    # convert strings to floats
    for row in range(len(data)):
        for column in range(3):
            data[row][column] = float(data[row][column])

    for row in range(len(data)):
        batches.append(row)
        fake.append(data[row][1])
        real.append(data[row][2])

    plt.plot(batches, fake, label='Fake prediction')
    plt.ylabel('Prediction (logits)')
    plt.xlabel('Minibatch')
    plt.plot(batches, real, label='Real prediction')
    plt.legend()
    plt.show()

# from latent_3d_points
#-----------------------------------------------------------------
def generator_noise_distribution(n_samples, ndims, mu, sigma):
    """
    Generate random noise
    """
    return np.random.normal(mu, sigma, (n_samples, ndims))
#-------------------------------------------------------------------

def generate(number_samples, noise_size, generator):
    """
    Generate random samples
    """
    z = generator_noise_distribution(number_samples, noise_size, mu=0, sigma=0.2)
    z_float = z.astype(np.float32)
    z_tensor = torch.from_numpy(z_float)
    z_tensor = z_tensor.to(device)
    out = generator(z_tensor)
    out_cpu = out.detach().cpu().numpy()
    print(out_cpu.shape)
    return out_cpu

def main(args):
    """
    Function that calls functions for generating and saving samples
    """
    samples = args.samples_to_generate
    noise_size = 128
    print(sys.argv)
    n_pc_points = 2048

    G = Point_Cloud_Generator(noise_size, [n_pc_points, 3])
    G = G.to(device)

    generator_checkpoint = torch.load(args.input_model)
    G.load_state_dict(generator_checkpoint['state_dict'])

    generated = generate(samples, noise_size, G)

    # from latent points
    #----------------------------------------------------------------------------------------
    for k in range(args.samples_to_generate):  # plot three (synthetic) random examples.
        plot_3d_point_cloud(generated[k][:, 0], generated[k][:, 1], generated[k][:, 2],
                            in_u_sphere=True)
    #-------------------------------------------------------------------------------------------

    if args.save_samples:
        if not os.path.exists("../data/"):
            os.makedirs("../data/")
        top_out_dir = '../data/'

        # from latent points, but modified
        synthetic_data_out_dir = os.path.join(top_out_dir, 'OUT/synthetic_samples/', 'wgan_paper_basic_pointnet')
        np.savez(os.path.join(synthetic_data_out_dir, 'generated'), generated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN Demo')
    parser.add_argument("--samples_to_generate", default=10, type=int, help="Samples to generate after each epoch")
    parser.add_argument("--save_samples", default=False, type=bool, help="Samples to generate after each epoch")
    #parser.add_argument("--input_model", default='checkpoints_pointnet_final/generator/model_epoch_71.pth', type=str, help="Trained model to use")
    #parser.add_argument("--input_model", default='checkpoints_pointnetplusplus/generator/model_epoch_1.pth', type=str, help="Trained model to use")

    parser.add_argument("--input_model", default='trained_models/pipeline1_pointnet_good_results/generator/model_epoch_71.pth', type=str,
                        help="Trained model to use")

    #parser.add_argument("--input_model", default='trained_models/pipeline2_pointnetplusplus_diverges/generator/model_epoch_1.pth', type=str,
    #                    help="Trained model to use")


    #parser.add_argument("--min_batches", default=135, type=int, help="Samples to generate after each epoch")
    #parser.add_argument("--discriminator_updates", default=5, type=int, help="Samples to generate after each epoch")
    args = parser.parse_args()
    main(args)

    #plot_loss("plots_pointnet_relu_final/wgan_training_loss.csv", 135, 5)
    #plot_predictions("plots_pointnet_relu_final/wgan_training_predictions.csv")

    #plot_loss("plots_pointnetplusplus5updates/wgan_training_loss.csv", 677, 5)
    #plot_predictions("plots_pointnetplusplus5updates/wgan_training_predictions.csv")

    #plot_loss("plots_pointnetplusplus1update/wgan_training_loss.csv", 677, 1)
    #plot_predictions("plots_pointnetplusplus1update/wgan_training_predictions.csv")

    #plot_loss("plots_reversewgan15/rwgan_training_loss.csv", 677, 1)
    #plot_predictions("plots_reversewgan15/rwgan_training_predictions.csv")