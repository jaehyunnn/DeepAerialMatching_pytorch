from __future__ import print_function, division
import os
import torch
import torchvision
from torch.autograd import Variable
from skimage import io
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf, SynthPairTnf_pck

class GoogleEarthPCK(Dataset):
    def __init__(self, csv_file, test_image_path, output_size=(540, 540), transform=None):
        self.out_h, self.out_w = output_size
        self.test_data = pd.read_csv(csv_file)
        self.src_names = self.test_data.iloc[:, 0]
        self.trg_names = self.test_data.iloc[:, 1]
        self.src_point_coords = self.test_data.iloc[:, 2:42].values.astype('float')
        self.theta_GT = self.test_data.iloc[:, 42:].values.astype('float')
        self.test_image_path = test_image_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        # get pre-processed images
        src_image, src_im_size = self.get_image(self.src_names, idx)
        trg_image, trg_im_size = self.get_image(self.trg_names, idx)

        # get pre-processed point coords
        src_point_coords = self.get_points(self.src_point_coords, idx)
        theta_GT = self.get_theta(self.theta_GT, idx)

        L_pck = torch.FloatTensor([torch.max(src_im_size)])

        sample = {'source_image': src_image, 'target_image': trg_image,
                  'source_im_size': src_im_size, 'target_im_size': trg_im_size,
                  'source_points': src_point_coords, 'theta_GT': theta_GT, 'L_pck': L_pck}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.test_image_path, img_name_list[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 20)

        #        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
        #        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords

    def get_theta(self, theta_list, idx):
        theta_GT = theta_list[idx, :].reshape(2, 3)
        theta_GT = torch.Tensor(theta_GT.astype(np.float32))
        return theta_GT

class GoogleEarthPCK_v2(Dataset):
    def __init__(self, csv_file, test_image_path, output_size=(240, 240), transform=None):
        self.out_h, self.out_w = output_size
        self.test_data = pd.read_csv(csv_file)
        self.src_names = self.test_data.iloc[:, 0]
        self.trg_names = self.test_data.iloc[:, 1]
        self.src_point_coords = self.test_data.iloc[:, 2:42].values.astype('float')
        self.trg_point_coords = self.test_data.iloc[:, 42:].values.astype('float')
        self.test_image_path = test_image_path
        self.transform = transform
        # no cuda as dataset is called from CPU threads in dataloader and produces confilct
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, idx):
        # get pre-processed images
        src_image, src_im_size = self.get_image(self.src_names, idx)
        trg_image, trg_im_size = self.get_image(self.trg_names, idx)

        # get pre-processed point coords
        src_point_coords = self.get_points(self.src_point_coords, idx)
        trg_point_coords = self.get_points(self.trg_point_coords, idx)

        sample = {'source_image': src_image, 'target_image': trg_image,
                  'source_im_size': src_im_size, 'target_im_size': trg_im_size,
                  'source_points': src_point_coords, 'target_points': trg_point_coords}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_image(self, img_name_list, idx):
        img_name = os.path.join(self.test_image_path, img_name_list[idx])
        image = io.imread(img_name)

        # get image size
        im_size = np.asarray(image.shape)

        # convert to torch Variable
        image = np.expand_dims(image.transpose((2, 0, 1)), 0)
        image = torch.Tensor(image.astype(np.float32))
        image_var = Variable(image, requires_grad=False)

        # Resize image using bilinear sampling with identity affine tnf
        image = self.affineTnf(image_var).data.squeeze(0)

        im_size = torch.Tensor(im_size.astype(np.float32))

        return (image, im_size)

    def get_points(self, point_coords_list, idx):
        point_coords = point_coords_list[idx, :].reshape(2, 20)

        #        # swap X,Y coords, as the the row,col order (Y,X) is used for computations
        #        point_coords = point_coords[[1,0],:]

        # make arrays float tensor for subsequent processing
        point_coords = torch.Tensor(point_coords.astype(np.float32))
        return point_coords
