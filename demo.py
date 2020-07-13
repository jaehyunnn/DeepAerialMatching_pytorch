from __future__ import print_function, division

from torch.autograd import Variable
from torchvision.transforms import Normalize

import torch
from model.AerialNet import net_single_stream as net
from image.normalization import NormalizeImageDict, normalize_image
from util.checkboard import createCheckBoard
from geotnf.transformation import GeometricTnf, theta2homogeneous
from geotnf.point_tnf import *
from util.torch_util import print_info
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
import warnings

import pickle
from functools import partial

import time


warnings.filterwarnings('ignore')

# torch.cuda.set_device(1) # Using second GPU

### Parameter
feature_extraction_cnn = 'se_resnext101'
model_path = 'trained_models/checkpoint.pth.tar'

source_image_path='failure_cases/source_438.jpg'
target_image_path='failure_cases/target_438.jpg'

### Load models
use_cuda = torch.cuda.is_available()

# Create model
print('Creating CNN model...')
model = net(use_cuda=use_cuda, geometric_model='affine', feature_extraction_cnn=feature_extraction_cnn)

pickle.load = partial(pickle.load, encoding="latin1")
pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

# Load trained weights
print('Loading trained model weights...')
checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
print("Reloading from--[%s]" % model_path)


### Load and preprocess images
resize = GeometricTnf(out_h=240, out_w=240, use_cuda=False)
normalizeTnf = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def Im2Tensor(image):
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    if use_cuda:
        image_var = image_var.cuda()
    return image_var

def preprocess_image(image):
    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32) / 255.0)
    image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resize(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var

source_image = io.imread(source_image_path)
target_image = io.imread(target_image_path)

source_image_var = preprocess_image(source_image)
target_image_var = preprocess_image(target_image)
target_image = np.float32(target_image/255.)

if use_cuda:
    source_image_var = source_image_var.cuda()
    target_image_var = target_image_var.cuda()

### Create image transformers
affTnf = GeometricTnf(geometric_model='affine', out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda=use_cuda)

batch = {'source_image': source_image_var, 'target_image':target_image_var}

resizeTgt = GeometricTnf(out_h=target_image.shape[0], out_w=target_image.shape[1], use_cuda = use_cuda)

### Evaluate model
model.eval()

start_time = time.time()
# Evaluate models
"""1st Affine"""
theta_aff, theta_aff_inv = model(batch)

# Calculate theta_aff_2
batch_size = theta_aff.size(0)
theta_aff_inv = theta_aff_inv.view(-1, 2, 3)
theta_aff_inv = torch.cat((theta_aff_inv, (torch.Tensor([0, 0, 1]).to('cuda').unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)
theta_aff_2 = theta_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

theta_aff_ensemble = (theta_aff + theta_aff_2) / 2  # Ensemble

### Process result
warped_image_aff = affTnf(Im2Tensor(source_image), theta_aff_ensemble.view(-1,2,3))
result_aff_np = warped_image_aff.squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
io.imsave('results/aff.jpg', result_aff_np)

"""2nd Affine"""
# Preprocess source_image_2
source_image_2 = normalize_image(resize(warped_image_aff.cpu()))
if use_cuda:
    source_image_2 = source_image_2.cuda()
theta_aff_aff, theta_aff_aff_inv = model({'source_image': source_image_2, 'target_image':batch['target_image']})

# Calculate theta_aff_2
batch_size = theta_aff_aff.size(0)
theta_aff_aff_inv = theta_aff_aff_inv.view(-1, 2, 3)
theta_aff_aff_inv = torch.cat((theta_aff_aff_inv, (torch.Tensor([0, 0, 1]).to('cuda').unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1)
theta_aff_aff_2 = theta_aff_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

theta_aff_aff_ensemble = (theta_aff_aff + theta_aff_aff_2) / 2  # Ensemble

theta_aff_ensemble = theta2homogeneous(theta_aff_ensemble)
theta_aff_aff_ensemble = theta2homogeneous(theta_aff_aff_ensemble)

theta = torch.bmm(theta_aff_aff_ensemble, theta_aff_ensemble).view(-1, 9)[:, :6]

### Process result
warped_image_aff_aff = affTnf(Im2Tensor(source_image), theta.view(-1,2,3))
result_aff_aff_np = warped_image_aff_aff.squeeze(0).transpose(0,1).transpose(1,2).cpu().detach().numpy()
io.imsave('results/aff_aff.jpg', result_aff_aff_np)

print()
print_info("# ====================================== #\n"
           "#            <Execution Time>            #\n"
           "#            - %.4s seconds -            #"%(time.time() - start_time)+"\n"
           "# ====================================== #",['yellow','bold'])

# Create overlay
aff_overlay = cv2.addWeighted(src1=result_aff_np, alpha= 0.4, src2=target_image, beta=0.8, gamma=0)
io.imsave('results/aff_overlay.jpg', np.clip(aff_overlay,-1,1))

# Create checkboard
aff_checkboard = createCheckBoard(result_aff_np, target_image)
io.imsave('results/aff_checkboard.jpg', aff_checkboard)

### Display
fig, axs = plt.subplots(2,3)
axs[0][0].imshow(source_image)
axs[0][0].set_title('Source')
axs[0][1].imshow(target_image)
axs[0][1].set_title('Target')
axs[0][2].imshow(result_aff_np)
axs[0][2].set_title('Affine')

axs[1][0].imshow(result_aff_aff_np)
axs[1][0].set_title('Affine X 2')
axs[1][1].imshow(aff_checkboard)
axs[1][1].set_title('Affine Checkboard')
axs[1][2].imshow(aff_overlay)
axs[1][2].set_title('Affine Overlay')

for i in range(2):
    for j in range(3):
        axs[i][j].axis('off')
fig.set_dpi(300)
plt.show()

