from __future__ import print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from geotnf.point_tnf import PointTnf

class TransformedGridLoss(nn.Module):
    def __init__(self, geometric_model='affine', use_cuda=True, grid_size=20):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model

        # define virtual grid of points to be transformed (grid_size x grid_size)
        axis_coords = np.linspace(-1,1,grid_size)
        self.N = grid_size*grid_size
        X,Y = np.meshgrid(axis_coords,axis_coords)
        X = np.reshape(X,(1,1,self.N))
        Y = np.reshape(Y,(1,1,self.N))
        P = np.concatenate((X,Y),1)
        self.P = Variable(torch.FloatTensor(P),requires_grad=False)
        self.pointTnf = PointTnf(use_cuda)
        if use_cuda:
            self.P = self.P.cuda()

    def forward(self, theta_AB, theta_BA, theta_AC, theta_CA, theta_GT_AB):
        # expand grid according to batch size
        batch_size = theta_AB.size()[0]
        P = self.P.expand(batch_size, 2, self.N)

        if self.geometric_model == 'affine':
            theta_GT_mat_AB = theta_GT_AB.view(-1, 2, 3)
            # inverse GT batch
            theta_GT_mat_temp = torch.cat((theta_GT_mat_AB, (torch.cuda.FloatTensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)
            for i in range(batch_size):
                theta_GT_mat_temp[i] = theta_GT_mat_temp[i].inverse()
            theta_GT_BA = theta_GT_mat_temp.view(-1,9)[:, :6]

            # compute transformed grid points using estimated and GT tnfs
            P_prime_GT = self.pointTnf.affPointTnf(theta_GT_AB, P)
            P_prime_GT_inv = self.pointTnf.affPointTnf(theta_GT_BA, P)

            P_prime_original = self.pointTnf.affPointTnf(theta_AB, P)
            P_prime_original_inv = self.pointTnf.affPointTnf(theta_BA, P)

            P_prime_jittered = self.pointTnf.affPointTnf(theta_AC, P)
            P_prime_jittered_inv = self.pointTnf.affPointTnf(theta_CA, P)


        # compute MSE loss on transformed grid points
        alpha = 0.5
        beta = 0.3
        gamma = 0.2

        l_original = torch.sum(torch.pow(P_prime_original - P_prime_GT,2),1) + torch.sum(torch.pow(P_prime_original_inv - P_prime_GT_inv,2),1)
        l_jittered = torch.sum(torch.pow(P_prime_jittered - P_prime_GT,2),1) + torch.sum(torch.pow(P_prime_jittered_inv - P_prime_GT_inv,2),1)
        l_identity = torch.sum(torch.pow(P_prime_original - P_prime_jittered,2),1) + torch.sum(torch.pow(P_prime_original_inv - P_prime_jittered_inv,2),1)

        Loss = (alpha*l_original) + (beta*l_jittered) + (gamma*l_identity)
        Loss = torch.mean(Loss)

        return Loss
