
import os
import glob
import torch.utils.data
import trimesh
import numpy as np
from matplotlib import pyplot as plt
from new_generator_discriminator import *
from torch.utils.data import dataloader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
#from pointnet import *
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

DATA_DIR = '../data/modelnet10/ModelNet10/'



# from https://keras.io/examples/vision/pointnet/
#------------------------------------------------------------------------------------------
def test(points):
    """
    Plots a point cloud
    """
    # plot points with predicted class and label
    fig = plt.figure(figsize=(15, 10))
    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, projection="3d")
        ax.scatter(points[i, :, 0], points[i, :, 1], points[i, :, 2])
        ax.set_axis_off()
    plt.show()


def augment(points, label):
    """
    Jitters points in point cloud
    """
    # jitter points
    points += np.random.uniform(-0.005, 0.005, points.shape)
    return points

def parse_dataset(num_points=2048):
    """
    Read point clouds from folders
    """

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    print(len(folders))

    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))
        print("test files: ", len(test_files))
        print("i: ", i)

        #for f in train_files:
        #for f in range(106):
        #    file = train_files[f]
        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    print("test_labels: ", test_labels)

    # added this
    np.save("../data/modelnet10/train_points2.npy", train_points)
    np.save("../data/modelnet10/test_points2.npy", test_points)
    np.save("../data/modelnet10/train_labels2.npy", train_labels)
    np.save("../data/modelnet10/test_labels2.npy", test_labels)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
    )

#-------------------------------------------------------------------------------------

# begin my code

# referenced hw4 for help making a data set
class CloudDataset(Dataset):
    """
    Custom data set for loading points and labels
    """
    def __init__(self, clouds, labels):
        self.clouds = clouds
        self.labels = labels

    def __len__(self):
        return len(self.clouds)

    def __getitem__(self, idx):
        label = self.labels[idx]
        cloud = self.clouds[idx]
        sample = {"Cloud": cloud, "Class": label}
        return sample


def train_pointnet(load_data=False):
    """
    Used for training pointnet classifier, as a benchmark - only use that discriminator with this function
    """
    # Top-dir of where point-clouds are stored.
    top_in_dir = '../data/modelnet10/ModelNet10/'

    n_pc_points = 2048
    num_classes = 10
    batch_size = 32
    num_epochs = 60

    if load_data:
        train_points, test_points, train_labels, test_labels = parse_dataset(n_pc_points)
    else:
        train_points = np.load("../data/modelnet10/train_points2.npy")
        test_points = np.load("../data/modelnet10/test_points2.npy")
        train_labels = np.load("../data/modelnet10/train_labels2.npy")
        test_labels = np.load("../data/modelnet10/test_labels2.npy")

    print("train points shape: ", train_points.shape)
    print("train labels shape: ", train_labels.shape)
    print("test points shape: ", test_points.shape)
    print("test labels shape: ", test_labels.shape)

    train_points = train_points.astype(np.float32)
    test_points = test_points.astype(np.float32)

    augmentedTrainPoints = augment(train_points, train_labels)

    X = torch.from_numpy(augmentedTrainPoints)
    Y = torch.from_numpy(test_points)

    print("X shape: ", X.shape)
    print(math.ceil(len(X) / batch_size))

    #X = X.cuda()
    #Y = Y.cuda()

    x_labels = torch.from_numpy(train_labels)#.cuda()
    y_labels = torch.from_numpy(test_labels)#.cuda()

    train_dataset = CloudDataset(X, x_labels)
    test_dataset = CloudDataset(Y, y_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #D = PointNetCls(10, False)
    D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], 10)
    #D = MLP_Discriminator_Paper([n_pc_points, 3], 10)
    # D = #PointNet_Discriminator1([n_pc_points, 3], out_channels=1)
    # D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], out_channels=1)
    D.filename = 'pointnet2_wgan_critic.to'

    D = D.to(device)

    optimizer = optim.Adam(D.parameters(), lr=0.05, betas=(0.9, 0.999))
    D.cuda()

    class_counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for label in train_labels:
        class_counts[label] = class_counts[label] + 1

    weights = 1 - (class_counts / len(train_labels))
    print("weights: ", weights)

    # https: // discuss.pytorch.org / t / passing - the - weights - to - crossentropyloss - correctly / 14731
    class_weights = torch.FloatTensor(weights).cuda()

    loss_function = nn.NLLLoss(weight=class_weights)

    for epoch in range(num_epochs):

        train_loss = 0

        for i, batch in enumerate(train_dataloader):
            print(i, batch["Cloud"].shape)
            #print("batch: ", batch)
            clouds = batch["Cloud"].to(device)
            #print("shape of clouds: ", clouds.shape)
            labels = batch["Class"].type(torch.LongTensor)
            #print("shape of labels: ", labels.shape)
            labels = labels.to(device)

            #print("clouds: ", clouds)
            clouds = clouds.transpose(2, 1)
            #outputs, probabilities = D(clouds)
            predictions, _, _ = D(clouds)
            #print("probabilities: ", probabilities[0])
            #print("outputs: ", outputs[0])
            #print("labels: ", labels[0])
            optimizer.zero_grad()

            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            #train_accuracy.append( predictions(labels.data.cpu().numpy(), outputs.data.cpu().numpy()))

            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, math.ceil(len(X) / batch_size), loss.item()))
            print()

    path = "../data/modelnet10/classifier_net.pth"
    torch.save(D.state_dict(), path)

    total = 0
    correct = 0

    D.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            clouds = batch["Cloud"].to(device)
            labels = batch["Class"].type(torch.LongTensor)
            labels = labels.to(device)

            clouds = clouds.transpose(2, 1)

            #outputs, probabilities = D(clouds)
            outputs, _, _ = D(clouds)

            print("outputs.data: ", outputs.data.shape)

            _, predicted = torch.max(outputs.data, 1)

            print("predicted: ", predicted)
            print("labels: ", labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("correct: ", correct)
    print("total: ", total)

    accuracy = correct / total
    print("Accuracy: ", accuracy)

    return D, dataloader, test_dataloader

def train_gan_discriminator(load_data=False):
    """
    Used for training one of the discriminators of the GAN for classifying on ModelNet10
    """
    # Top-dir of where point-clouds are stored.
    top_in_dir = '../data/modelnet10/ModelNet10/'

    n_pc_points = 2048
    num_classes = 10
    batch_size = 32
    num_epochs = 40

    if load_data:
        train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(n_pc_points)
    else:
        train_points = np.load("../data/modelnet10/train_points2.npy")
        test_points = np.load("../data/modelnet10/test_points2.npy")
        train_labels = np.load("../data/modelnet10/train_labels2.npy")
        test_labels = np.load("../data/modelnet10/test_labels2.npy")

    print("train points shape: ", train_points.shape)
    print("train labels shape: ", train_labels.shape)
    print("test points shape: ", test_points.shape)
    print("test labels shape: ", test_labels.shape)

    train_points = train_points.astype(np.float32)
    test_points = test_points.astype(np.float32)

    augmentedTrainPoints = augment(train_points, train_labels)

    X = torch.from_numpy(augmentedTrainPoints)
    Y = torch.from_numpy(test_points)

    print("X shape: ", X.shape)
    print(math.ceil(len(X) / batch_size))

    #X = X.cuda()
    #Y = Y.cuda()

    x_labels = torch.from_numpy(train_labels)#.cuda()
    y_labels = torch.from_numpy(test_labels)#.cuda()

    train_dataset = CloudDataset(X, x_labels)
    test_dataset = CloudDataset(Y, y_labels)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    #D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], 10)
    D = MLP_Discriminator_Paper([n_pc_points, 3], 10)
    # D = #PointNet_Discriminator1([n_pc_points, 3], out_channels=1)
    # D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], out_channels=1)
    D.filename = 'pointnet2_wgan_critic.to'

    D = D.to(device)

    optimizer = optim.Adam(D.parameters(), lr=0.001, betas=(0.5, 0.9))
    D.cuda()

    class_counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for label in train_labels:
        class_counts[label] = class_counts[label] + 1

    weights = 1 - (class_counts / len(train_labels))
    print("weights: ", weights)

    # https: // discuss.pytorch.org / t / passing - the - weights - to - crossentropyloss - correctly / 14731
    class_weights = torch.FloatTensor(weights).cuda()

    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    # from assignment 2
    for epoch in range(num_epochs):

        train_loss = 0
        #train_accuracy = []

        for i, batch in enumerate(train_dataloader):
            print(i, batch["Cloud"].shape)
            #print("batch: ", batch)
            clouds = batch["Cloud"].to(device)
            #print("shape of clouds: ", clouds.shape)
            labels = batch["Class"].type(torch.LongTensor)
            #print("shape of labels: ", labels.shape)
            labels = labels.to(device)

            #print("clouds: ", clouds)
            outputs = D(clouds, None)
            #print("probabilities: ", probabilities[0])
            #print("outputs: ", outputs[0])
            #print("labels: ", labels[0])
            optimizer.zero_grad()

            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            #train_accuracy.append( predictions(labels.data.cpu().numpy(), outputs.data.cpu().numpy()))

            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, math.ceil(len(X) / batch_size), loss.item()))
            print()

    path = "../data/modelnet10/classifier_net.pth"
    torch.save(D.state_dict(), path)

    total = 0
    correct = 0

    D.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            clouds = batch["Cloud"].to(device)
            labels = batch["Class"].type(torch.LongTensor)
            labels = labels.to(device)

            outputs = D(clouds, None)

            print("outputs.data: ", outputs.data.shape)

            _, predicted = torch.max(outputs.data, 1)

            print("predicted: ", predicted)
            print("labels: ", labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("correct: ", correct)
    print("total: ", total)

    accuracy = correct / total
    print("Accuracy: ", accuracy)

    return D, dataloader, test_dataloader

def testClassifier(D, dataLoader, test_dataLoader, n_pc_points=2048):
    """
    Testing without training first
    """
    path = "../data/modelnet10/classifier_net.pth"

    D = PointNet_Plus_Plus_Discriminator([n_pc_points, 3], 10)
    #D = MLP_Discriminator_Paper([n_pc_points, 3], 10)
    D.load_state_dict(torch.load(path))
    D.eval()
    D.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_dataLoader:
            clouds = batch["Cloud"].to(device)
            labels = batch["Class"].type(torch.LongTensor)
            labels = labels.to(device)

            outputs = D(clouds)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct/total
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    #D, dataloader, testloader = train_pointnet(False)
    D, dataloader, testloader = train_gan_discriminator(True)
    #testClassifier(D, dataloader, testloader)