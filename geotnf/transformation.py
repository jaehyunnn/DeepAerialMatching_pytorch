from __future__ import print_function, division
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.autograd import Variable
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
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b, 2, 3)
            theta_batch = Variable(theta_batch, requires_grad=False)

        sampling_grid = self.gridGen(theta_batch)

        # rescale grid according to crop_factor and padding_factor
        sampling_grid.data = sampling_grid.data * padding_factor * crop_factor
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)

        return warped_image_batch


class SynthPairTnf(object):
    """

    Generate a synthetically warped training pair using an affine transformation.

    """

    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w,
                                         use_cuda=self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w,
                                         use_cuda=self.use_cuda)

    def __call__(self, batch):
        src_image_batch, trg_image_batch, trg_image_jit_batch, theta_batch = batch['src_image'], batch['trg_image'], batch['trg_image_jit'], batch['theta']

        if self.use_cuda:
            src_image_batch = src_image_batch.cuda()
            trg_image_batch = trg_image_batch.cuda()
            trg_image_jit_batch = trg_image_jit_batch.cuda()
            theta_batch = theta_batch.cuda()

        b, c, h, w = src_image_batch.size()

        # generate symmetrically padded image for bigger sampling region
        src_image_batch = self.symmetricImagePad(src_image_batch, self.padding_factor)
        trg_image_batch = self.symmetricImagePad(trg_image_batch, self.padding_factor)
        trg_image_jit_batch = self.symmetricImagePad(trg_image_jit_batch, self.padding_factor)

        # convert to variables
        src_image_batch = Variable(src_image_batch, requires_grad=False)
        trg_image_batch = Variable(trg_image_batch, requires_grad=False)
        trg_image_jit_batch = Variable(trg_image_jit_batch, requires_grad=False)
        theta_batch = Variable(theta_batch, requires_grad=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(src_image_batch, None, self.padding_factor, self.crop_factor)  # Identity is used as no theta given
        # get transformed image
        warped_image_batch = self.geometricTnf(trg_image_batch, theta_batch, self.padding_factor, self.crop_factor)  # Identity is used as no theta given
        warped_image_jit_batch = self.geometricTnf(trg_image_jit_batch, theta_batch, self.padding_factor, self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch, 'target_image_jit':warped_image_jit_batch, 'theta_GT': theta_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)
        return image_batch

class SynthPairTnf_pck(object):
    """

    Generate a synthetically warped test pair using an affine transformation for pck-evaluation

    """

    def __init__(self, use_cuda=True, geometric_model='affine', crop_factor=9 / 16, output_size=(240, 240),
                 padding_factor=0.5):
        assert isinstance(use_cuda, (bool))
        assert isinstance(crop_factor, (float))
        assert isinstance(output_size, (tuple))
        assert isinstance(padding_factor, (float))
        self.use_cuda = use_cuda
        self.crop_factor = crop_factor
        self.padding_factor = padding_factor
        self.out_h, self.out_w = output_size
        self.rescalingTnf = GeometricTnf('affine', self.out_h, self.out_w,
                                         use_cuda=self.use_cuda)
        self.geometricTnf = GeometricTnf(geometric_model, self.out_h, self.out_w,
                                         use_cuda=self.use_cuda)

    def __call__(self, batch):
        src_image_batch, trg_image_batch, theta_batch = batch['src_image'], batch['trg_image'], batch['theta']

        # generate symmetrically padded image for bigger sampling region
        src_image_batch = self.symmetricImagePad(src_image_batch, self.padding_factor)
        trg_image_batch = self.symmetricImagePad(trg_image_batch, self.padding_factor)

        # convert to variables
        src_image_batch = Variable(src_image_batch, requires_grad=False)
        trg_image_batch = Variable(trg_image_batch, requires_grad=False)
        theta_batch = Variable(theta_batch, requires_grad=False)

        # get cropped image
        cropped_image_batch = self.rescalingTnf(src_image_batch, None, self.padding_factor, self.crop_factor)  # Identity is used as no theta given
        # get transformed image
        warped_image_batch = self.geometricTnf(trg_image_batch, theta_batch, self.padding_factor, self.crop_factor)  # Identity is used as no theta given

        return {'source_image': cropped_image_batch, 'target_image': warped_image_batch}

    def symmetricImagePad(self, image_batch, padding_factor):
        b, c, h, w = image_batch.size()
        pad_h, pad_w = int(h * padding_factor), int(w * padding_factor)
        idx_pad_left = torch.LongTensor(range(pad_w - 1, -1, -1))
        idx_pad_right = torch.LongTensor(range(w - 1, w - pad_w - 1, -1))
        idx_pad_top = torch.LongTensor(range(pad_h - 1, -1, -1))
        idx_pad_bottom = torch.LongTensor(range(h - 1, h - pad_h - 1, -1))
        if self.use_cuda:
            idx_pad_left = idx_pad_left.cuda()
            idx_pad_right = idx_pad_right.cuda()
            idx_pad_top = idx_pad_top.cuda()
            idx_pad_bottom = idx_pad_bottom.cuda()
        image_batch = torch.cat((image_batch.index_select(3, idx_pad_left), image_batch,
                                 image_batch.index_select(3, idx_pad_right)), 3)
        image_batch = torch.cat((image_batch.index_select(2, idx_pad_top), image_batch,
                                 image_batch.index_select(2, idx_pad_bottom)), 2)
        return image_batch


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
        return F.affine_grid(theta, out_size)

def theta2homogeneous(theta):
    batch_size = theta.size(0)
    theta = theta.view(-1, 2, 3)
    theta = torch.cat((theta, (torch.Tensor([0, 0, 1]).to('cuda').unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)

    return theta