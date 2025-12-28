from __future__ import annotations

import torch
import torch.nn as nn
from transforms.point import PointTnf


class TransformedGridLoss(nn.Module):
    """Grid-based loss for geometric transformation learning.

    Computes loss by comparing transformed grid points between predicted
    and ground truth transformations.
    """

    def __init__(
        self,
        geometric_model: str = 'affine',
        grid_size: int = 20,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        super(TransformedGridLoss, self).__init__()
        self.geometric_model = geometric_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # define virtual grid of points to be transformed (grid_size x grid_size)
        axis_coords = torch.linspace(-1, 1, grid_size)
        self.N = grid_size * grid_size
        Y, X = torch.meshgrid(axis_coords, axis_coords, indexing='ij')
        X = X.reshape(1, 1, self.N)
        Y = Y.reshape(1, 1, self.N)
        P = torch.cat((X, Y), dim=1)
        self.register_buffer('P', P)
        self.pointTnf = PointTnf()

    def forward(self, theta_AB, theta_BA, theta_AC, theta_CA, theta_GT_AB):
        # expand grid according to batch size
        batch_size = theta_AB.size()[0]
        device = theta_AB.device
        P = self.P.to(device).expand(batch_size, 2, self.N).contiguous()

        if self.geometric_model == 'affine':
            theta_GT_mat_AB = theta_GT_AB.reshape(-1, 2, 3)
            # inverse GT batch using batch matrix inverse
            homogeneous_row = torch.tensor([0, 0, 1], dtype=theta_GT_mat_AB.dtype, device=device)
            homogeneous_row = homogeneous_row.reshape(1, 1, 3).expand(batch_size, 1, 3).contiguous()
            theta_GT_mat_temp = torch.cat((theta_GT_mat_AB, homogeneous_row), dim=1)
            theta_GT_mat_inv = torch.linalg.inv(theta_GT_mat_temp)
            theta_GT_BA = theta_GT_mat_inv[:, :2, :].reshape(-1, 6)

            # compute transformed grid points using estimated and GT tnfs
            P_prime_GT = self.pointTnf.affine_transform(theta_GT_AB, P)
            P_prime_GT_inv = self.pointTnf.affine_transform(theta_GT_BA, P)

            P_prime_original = self.pointTnf.affine_transform(theta_AB, P)
            P_prime_original_inv = self.pointTnf.affine_transform(theta_BA, P)

            P_prime_jittered = self.pointTnf.affine_transform(theta_AC, P)
            P_prime_jittered_inv = self.pointTnf.affine_transform(theta_CA, P)

        # compute MSE loss on transformed grid points
        l_original = torch.sum(torch.pow(P_prime_original - P_prime_GT, 2), 1) + \
                     torch.sum(torch.pow(P_prime_original_inv - P_prime_GT_inv, 2), 1)
        l_jittered = torch.sum(torch.pow(P_prime_jittered - P_prime_GT, 2), 1) + \
                     torch.sum(torch.pow(P_prime_jittered_inv - P_prime_GT_inv, 2), 1)
        l_identity = torch.sum(torch.pow(P_prime_original - P_prime_jittered, 2), 1) + \
                     torch.sum(torch.pow(P_prime_original_inv - P_prime_jittered_inv, 2), 1)

        loss = (self.alpha * l_original) + (self.beta * l_jittered) + (self.gamma * l_identity)
        loss = torch.mean(loss)

        return loss
