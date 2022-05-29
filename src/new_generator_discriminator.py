'''
Created on April 23, 2022

@author: Matthew Spahl
'''

import numpy as np
import torch
import torch.nn as nn
#import pointnet2_ops
import pointnet2_ops_lib.pointnet2_ops.pointnet2_modules as pointnet2

# generates point cloud from random noise
class Point_Cloud_Generator(torch.nn.Module):
    def __init__(self, z, pc_dims):
        super(Point_Cloud_Generator, self).__init__()

        self.n_points = pc_dims[0] # points per cloud
        self.in_channels = pc_dims[1] # dimensions of each point, 3

        self.fc1 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 512)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.batch_norm_3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        #self.fc3 = nn.Linear(128, 1024)
        #nn.init.xavier_uniform_(self.fc3.weight)
        #self.batch_norm_3 = nn.BatchNorm1d(1024)
        #self.relu3 = nn.ReLU()

        self.fc4 = nn.Linear(512, 1024)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.batch_norm_4 = nn.BatchNorm1d(1024)
        self.relu4 = nn.ReLU()

        #self.fc4 = nn.Linear(512, 2048 * 3)
        #nn.init.xavier_uniform_(self.fc4.weight)
        #self.batch_norm_4 = nn.BatchNorm1d(1024)
        #self.relu4 = nn.ReLU()

        self.fc5 = nn.Linear(1024, 2048 * 3)
        nn.init.xavier_uniform_(self.fc5.weight)
        #self.batch_norm_5 = nn.BatchNorm1d(2048 * 3)
        #self.relu5 = nn.ReLU()


    def forward(self, x):
        # z: [batch_size, latent_channels]

        print(x.shape)

        x = self.fc1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.batch_norm_3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.batch_norm_4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        out = x.contiguous().view(-1, self.n_points, 3)
        return out

# generates point cloud from random noise
class Point_Cloud_Generator_Short(torch.nn.Module):
    def __init__(self, z, pc_dims):
        super(Point_Cloud_Generator_Short, self).__init__()

        self.n_points = pc_dims[0] # points per cloud
        self.in_channels = pc_dims[1] # dimensions of each point, 3

        self.fc1 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.batch_norm_1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(128, 512)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.batch_norm_3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()

        #self.fc3 = nn.Linear(128, 1024)
        #nn.init.xavier_uniform_(self.fc3.weight)
        #self.batch_norm_3 = nn.BatchNorm1d(1024)
        #self.relu3 = nn.ReLU()

        #self.fc4 = nn.Linear(512, 1024)
        #nn.init.xavier_uniform_(self.fc4.weight)
        #self.batch_norm_4 = nn.BatchNorm1d(1024)
        #self.relu4 = nn.ReLU()

        self.fc4 = nn.Linear(512, 2048 * 3)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.batch_norm_4 = nn.BatchNorm1d(1024)
        self.relu4 = nn.ReLU()

        #self.fc5 = nn.Linear(1024, 2048 * 3)
        #nn.init.xavier_uniform_(self.fc5.weight)


    def forward(self, x):
        # z: [batch_size, latent_channels]

        x = self.fc1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.batch_norm_3(x)
        x = self.relu3(x)
        x = self.fc4(x)

        x = x.contiguous().view(-1, self.n_points, 3)
        return x


# implmentation of single scale grouping version of pointnet plus plus, based on specifications in paper
# batch norm removed, as instructed in wgan paper
class PointNet_Plus_Plus_Discriminator(torch.nn.Module):
    def __init__(self, in_signal, out_channels):
        super(PointNet_Plus_Plus_Discriminator, self).__init__()

        self.n_points = in_signal[0] # points per cloud, 2048
        self.in_channels = in_signal[1] # 3
        print("self in channels: ", self.in_channels)
        self.out_channels = out_channels

        self.SA1 = pointnet2.PointnetSAModule(mlp=[0, 64, 64, 128], npoint=512, radius=0.2, nsample=64, bn=True, use_xyz=True)
        self.SA2 = pointnet2.PointnetSAModule(mlp=[128, 128, 128, 256], npoint=128, radius=0.4, nsample=64, bn=True, use_xyz=False)
        self.SA3 = pointnet2.PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=False)

        self.fc1 = nn.Linear(1024, 512)
        #self.batch_norm_1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout_layer1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        #self.batch_norm_2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout_layer2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, self.out_channels)
        #self.softmax = nn.Softmax(1)

    def forward(self, x, features):
        # x: [batch_size, 3]

        x, features = self.SA1(x, features)
        x, features = self.SA2(x, features)
        empty_x, features = self.SA3(x, features)

        # https://pytorch.org/docs/stable/generated/torch.squeeze.html
        x = torch.squeeze(features)

        x = self.fc1(x)
        #x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout_layer1(x)

        x = self.fc2(x)
        #x = self.batch_norm_2(x)
        x = self.relu2(x)
        x = self.dropout_layer2(x)

        predictions = self.fc3(x)
        #print("shape of predictions: ", predictions.shape)
        #probabilities = self.softmax(predictions)

        return predictions #, probabilities

# implmentation of single scale grouping version of pointnet plus plus, based on specifications in paper
# used for testing with ModelNet 10, but not with the GAN, as batch norm re-enabled
class PointNet_Plus_Plus_Discriminator_Standard(torch.nn.Module):
    def __init__(self, in_signal, out_channels):
        super(PointNet_Plus_Plus_Discriminator, self).__init__()

        self.n_points = in_signal[0] # points per cloud, 2048
        self.in_channels = in_signal[1] # 3
        print("self in channels: ", self.in_channels)
        self.out_channels = out_channels

        self.SA1 = pointnet2.PointnetSAModule(mlp=[0, 64, 64, 128], npoint=512, radius=0.2, nsample=64, bn=True, use_xyz=True)
        self.SA2 = pointnet2.PointnetSAModule(mlp=[128, 128, 128, 256], npoint=128, radius=0.4, nsample=64, bn=True, use_xyz=False)
        self.SA3 = pointnet2.PointnetSAModule(mlp=[256, 256, 512, 1024], use_xyz=False)

        self.fc1 = nn.Linear(1024, 512)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout_layer1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout_layer2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, self.out_channels)
        #self.softmax = nn.Softmax(1)

    def forward(self, x, features):
        # x: [batch_size, 3]

        x, features = self.SA1(x, features)
        x, features = self.SA2(x, features)
        empty_x, features = self.SA3(x, features)

        # https://pytorch.org/docs/stable/generated/torch.squeeze.html
        x = torch.squeeze(features)

        x = self.fc1(x)
        x = self.batch_norm_1(x)
        x = self.relu1(x)
        x = self.dropout_layer1(x)
        x = self.fc2(x)
        x = self.batch_norm_2(x)

        x = self.relu2(x)
        x = self.dropout_layer2(x)
        predictions = self.fc3(x)
        #print("shape of predictions: ", predictions.shape)
        #probabilities = self.softmax(predictions)

        return predictions #, probabilities

# MLP module from assignment 3
def MLP(channels, enable_group_norm=False):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)
        return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i-1], channels[i]), nn.LeakyReLU(negative_slope=0.2), nn.GroupNorm(num_groups[i], channels[i]))
            for i in range(1, len(channels))])
    else:
        return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i-1], channels[i]), nn.LeakyReLU(negative_slope=0.2)) for i in range(1, len(channels))])

# MLP module from assignment 3 modified to use relu to match paper
def MLP_Relu(channels, enable_group_norm=False):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)
        return nn.Sequential(*[nn.Sequential(nn.Conv1d(channels[i-1], channels[i], kernel_size=1, stride=1), nn.ReLU(), nn.GroupNorm(num_groups[i], channels[i]))
            for i in range(1, len(channels))])
    else:
        return nn.Sequential(*[nn.Sequential(nn.Linear(channels[i-1], channels[i]), nn.ReLU()) for i in range(1, len(channels))])

# Pytorch implementation of the paper's discriminator (originally in tensorflow 1)
class MLP_Discriminator_Paper(torch.nn.Module):
    def __init__(self, in_signal, out_channels):
        super(MLP_Discriminator_Paper, self).__init__()

        self.mlp1 = MLP_Relu([3, 64, 128, 256, 512], False)

        self.fc1 = nn.Linear(512, 128)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(128, 64)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(64, out_channels)
        nn.init.xavier_uniform_(self.fc3.weight)

        #self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        # x: [batch_size, 3]

        x = self.mlp1(x)

        x = x.max(dim=1).values

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x

#-----------------------------------------------------------------------------------------


# regular pointnet from assignment 3 to test against pointnet plus plus
# not used
class PointNet_Discriminator1(torch.nn.Module):
    def __init__(self, in_signal, out_channels):
        super(PointNet_Discriminator1, self).__init__()

        self.mlp1 = MLP([3, 32, 64, 128], False)
        self.mlp2 = MLP([128, 128], False)

        self.mlp3 = MLP([256, 128, 64], False)

        self.linear = nn.Linear(64, 1)


        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        # x: [batch_size, 3]

        print("shape of x: ", x.shape)
        f_i = self.mlp1(x)

        print("shape of f_i: ", f_i.shape)

        h_i = self.mlp2(f_i)

        print("shape of h_i: ", h_i.shape)

        g = torch.max(h_i, 1).values

        print("g shape: ", g.shape)

        g = g.repeat(f_i.size(dim=0), f_i.size(dim=1), 1)

        print("h_i shape: ", h_i.shape)
        print("g shape: ", g.shape)

        concatenated = torch.cat((f_i, g), 2)
        concatenated = self.mlp3(concatenated)

        y_i = self.linear(concatenated)

        return y_i

# basic MLP discriminator based on one of the discriminators of the paper, didn't use this
class MLP_Discriminator(torch.nn.Module):
    def __init__(self, in_signal, out_channels):
        super(MLP_Discriminator, self).__init__()

        self.mlp1 = MLP([3, 32, 64, 128], False)
        self.mlp2 = MLP([128, 256], False)

        self.mlp3 = MLP([256, 128, 64], False)

        self.linear = nn.Linear(64, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, features):
        # x: [batch_size, 3]

        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        y_i = self.linear(x)

        return y_i