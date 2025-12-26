from __future__ import print_function, division
import numpy as np
import torch
from torch.nn.modules.module import Module
import torch.nn.functional as F


class GeometricTnf(object):
    """

    Geometric transfromation to an image batch (wrapped in a PyTorch Variable)
    ( can be used with no transformation to perform bilinear resizing )

    """

    def __init__(self, geometric_model='affine', out_h=240, out_w=240, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        if geometric_model == 'affine':
            self.gridGen = AffineGridGen(out_h, out_w)
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1, 0, 0], [0, 1, 0]]), 0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, padding_factor=1.0, crop_factor=1.0):
        b, c, h, w = image_batch.size()
        if theta_batch is None:
            theta_batch = self.theta_identity.expand(b, 2, 3).to(image_batch.device)

        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        sampling_grid = sampling_grid * padding_factor * crop_factor
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid, align_corners=False)

        return warped_image_batch


def symmetric_image_pad(image_batch, padding_factor):
    """Apply symmetric padding to image batch for larger sampling region."""
    b, c, h, w = image_batch.size()
    pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
    device = image_batch.device

    idx_pad_left = torch.arange(pad_w - 1, -1, -1, device=device)
    idx_pad_right = torch.arange(w - 1, w - pad_w - 1, -1, device=device)
    idx_pad_top = torch.arange(pad_h - 1, -1, -1, device=device)
    idx_pad_bottom = torch.arange(h - 1, h - pad_h - 1, -1, device=device)

    image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                             image_batch.index_select(3, idx_pad_right)), 3)
    image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                             image_batch.index_select(2, idx_pad_bottom)), 2)
    return image_batch


class SynthPairTnfBase(object):
    """Base class for synthetic pair transformation."""

    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9/16,
                 output_size=(240, 240), padding_factor=0.5):
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w, use_cuda=use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w, use_cuda=use_cuda)


class SynthPairTnf(SynthPairTnfBase):
    """Generate synthetically warped training pair with jittered target."""

    def __call__(self, batch):
        src_image_batch = batch['src_image'].to(self.device)
        trg_image_batch = batch['trg_image'].to(self.device)
        trg_image_jit_batch = batch['trg_image_jit'].to(self.device)
        theta_batch = batch['theta'].to(self.device)

        # generate symmetrically padded image for bigger sampling region
        src_image_batch = symmetric_image_pad(src_image_batch, self.padding_factor)
        trg_image_batch = symmetric_image_pad(trg_image_batch, self.padding_factor)
        trg_image_jit_batch = symmetric_image_pad(trg_image_jit_batch, self.padding_factor)

        # get cropped image (Identity is used as no theta given)
        cropped_image_batch = self.rescalingTnf(src_image_batch, None, self.padding_factor, self.crop_factor)
        # get transformed image
        warped_image_batch = self.geometricTnf(trg_image_batch, theta_batch, self.padding_factor, self.crop_factor)
        warped_image_jit_batch = self.geometricTnf(trg_image_jit_batch, theta_batch, self.padding_factor, self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch,
                'target_image_jit': warped_image_jit_batch, 'theta_GT': theta_batch}


class SynthPairTnf_pck(SynthPairTnfBase):
    """Generate synthetically warped test pair for PCK evaluation."""

    def __call__(self, batch):
        src_image_batch = batch['src_image']
        trg_image_batch = batch['trg_image']
        theta_batch = batch['theta']

        # generate symmetrically padded image for bigger sampling region
        src_image_batch = symmetric_image_pad(src_image_batch, self.padding_factor)
        trg_image_batch = symmetric_image_pad(trg_image_batch, self.padding_factor)

        # get cropped image (Identity is used as no theta given)
        cropped_image_batch = self.rescalingTnf(src_image_batch, None, self.padding_factor, self.crop_factor)
        # get transformed image
        warped_image_batch = self.geometricTnf(trg_image_batch, theta_batch, self.padding_factor, self.crop_factor)

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch}


class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch=3):
        super(AffineGridGen, self).__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch

    def forward(self, theta):
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size, self.out_ch, self.out_h, self.out_w))
        return F.affine_grid(theta, out_size, align_corners=False)

def theta2homogeneous(theta):
    """Convert 2x3 affine matrix to 3x3 homogeneous matrix."""
    batch_size = theta.size(0)
    device = theta.device
    theta = theta.view(-1, 2, 3)
    homogeneous_row = torch.tensor([[0, 0, 1]], dtype=theta.dtype, device=device)
    homogeneous_row = homogeneous_row.expand(batch_size, 1, 3)
    theta = torch.cat((theta, homogeneous_row), dim=1)
    return theta