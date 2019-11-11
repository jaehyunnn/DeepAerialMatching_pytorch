from __future__ import print_function, division
import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from model.AerialNet_two_stream import net
from data.pck_dataset import GoogleEarthPCK
from data.download import download_eval
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars, str_to_bool, print_info
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf, SynthPairTnf_pck
import numpy as np
from tqdm import tqdm

torch.cuda.set_device(1) # Using second GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    print_info('[Deep Aerial Matching] Evaluation Script', ['green', 'bold'])

    # Argument parser
    parser = argparse.ArgumentParser(description='Deep Aerial Matching PyTorch Implementation')
    # Paths
    parser.add_argument('--model-aff', type=str, default='trained_models/checkpoint_seresnext101.pth.tar', help='Trained affine model filename')
    parser.add_argument('--batch-size', type=int, default=16, help='Test batch size')
    parser.add_argument('--feature-extraction-cnn', type=str, default='se_resnext101', help='Feature extraction architecture')
    parser.add_argument('--image-path', type=str, default='datasets/GoogleEarth/GoogleEarth_pck', help='Path to PCK dataset')
    parser.add_argument('--dataset', type=str, default='GoogleEarth_pck', help='Select evaluation dataset')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # Create model
    print('Creating CNN model...')
    model = net(use_cuda=use_cuda, geometric_model='affine', feature_extraction_cnn=args.feature_extraction_cnn)

    # Load trained weights
    print('Loading trained model weights...')
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)

    # Load parameters of FeatureExtraction
    for name, param in model.FeatureExtraction.state_dict().items():
        model.FeatureExtraction.state_dict()[name].copy_(checkpoint['state_dict']['FeatureExtraction.' + name])
    # Load parameters of FeatureRegression (Affine)
    for name, param in model.FeatureRegression.state_dict().items():
        model.FeatureRegression.state_dict()[name].copy_(checkpoint['state_dict']['FeatureRegression.' + name])

    # Dataset and dataloader
    if args.dataset == 'GoogleEarth_pck':
        # Download dataset
        download_eval('datasets/GoogleEarth')
        dataset = GoogleEarthPCK(csv_file=os.path.join(args.image_path, 'test_pairs.csv'),
                                 test_image_path=args.image_path,
                                 transform=NormalizeImageDict(['source_image', 'target_image']))
    if use_cuda:
        batch_size = args.batch_size
    else:
        batch_size = 1

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    pair_generation_tnf = SynthPairTnf_pck(geometric_model='affine',use_cuda=use_cuda)

    batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

    # Instantiate point transformer
    pt = PointTnf(use_cuda=use_cuda)

    # Instatiate image transformers
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)


    # Compute PCK
    def correct_keypoints(source_points, warped_points, L_pck, tau=0.01):
        # compute correct keypoints
        point_distance = torch.pow(torch.sum(torch.pow(source_points - warped_points, 2), 1), 0.5).squeeze(1)
        L_pck_mat = torch.cuda.FloatTensor(L_pck).expand_as(point_distance)
        correct_points = torch.le(point_distance, L_pck_mat * tau)
        num_of_correct_points = torch.sum(correct_points)
        num_of_points = correct_points.numel()
        return (num_of_correct_points, num_of_points)


    print('Computing PCK...')
    print("[%s]"%args.model_aff)
    print_info("# ===================================== #\n"
               "\t\t\t...Eval config...\n"
               "\t\t\t------------------\n"
               "\t\t CNN model: " + args.feature_extraction_cnn + "\n"
               "\t\t # of eval data: " + str(len(dataset)) + "\n\n"
               "\t\t Batch size: " + str(args.batch_size) + "\n"
               "# ===================================== #\n", ['yellow', 'bold'])

    total_correct_points_aff_1 = 0
    total_correct_points_aff_aff_1 = 0
    total_points_1 = 0

    total_correct_points_aff_3 = 0
    total_correct_points_aff_aff_3 = 0
    total_points_3 = 0

    total_correct_points_aff_5 = 0
    total_correct_points_aff_aff_5 = 0
    total_points_5 = 0
    for i, batch in enumerate(tqdm(dataloader)):

        batch = batchTensorToVars(batch)

        source_im_size = batch['source_im_size']
        target_im_size = batch['target_im_size']

        batch_tnf = pair_generation_tnf({'src_image':batch['source_image'], 'trg_image':batch['target_image'], 'theta':batch['theta_GT'] })

        batch['source_image'] = batch_tnf['source_image']
        batch['target_image'] = batch_tnf['target_image']

        source_points = batch['source_points']
        theta_GT = batch['theta_GT']

        # warp points with estimated transformations
        source_points_norm = PointsToUnitCoords(source_points, source_im_size/2)  # Normalize image coordinate

        model.eval()
        """ 1st Affine """
        theta_aff, theta_aff_inv = model(batch)

        # Calculate theta_aff_2 for Ensemble
        batch_size = theta_aff.size(0)
        theta_aff_inv = theta_aff_inv.view(-1, 2, 3)
        theta_aff_inv = torch.cat((theta_aff_inv, (torch.Tensor([0, 0, 1]).to(device).unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))), 1) # Convert to homogeneous form
        theta_aff_2 = theta_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

        a = 0.5
        b= (1-a)
        theta_aff_ensemble = (a*theta_aff) + (b*theta_aff_2)

        # do affine
        warped_points_aff_norm_GT = pt.affPointTnf(theta_GT, source_points_norm)
        warped_points_aff_norm = pt.affPointTnf(theta_aff_ensemble, source_points_norm)

        warped_points_aff_GT = PointsToPixelCoords(warped_points_aff_norm_GT, source_im_size/2)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, source_im_size/2)  # Un-normalize

        """ 2nd Affine """
        warped_image_aff = affTnf(batch['source_image'], theta_aff_ensemble.view(-1, 2, 3))
        theta_aff_aff, theta_aff_aff_inv = model({'source_image': warped_image_aff, 'target_image': batch['target_image']})

        # Calculate theta_aff_aff_2
        batch_size = theta_aff_aff.size(0)
        theta_aff_aff_inv = theta_aff_aff_inv.view(-1, 2, 3)
        theta_aff_aff_inv = torch.cat((theta_aff_aff_inv, (torch.Tensor([0, 0, 1]).to(device).unsqueeze(0).unsqueeze(1).expand(batch_size, 1, 3))),1)
        theta_aff_aff_2 = theta_aff_aff_inv.inverse().contiguous().view(-1, 9)[:, :6]

        theta_aff_aff_ensemble = (a*theta_aff_aff) + (b*theta_aff_aff_2)

        # do affine X 2
        warped_points_aff_aff_norm = pt.affPointTnf(theta_aff_aff_ensemble, source_points_norm)
        warped_points_aff_aff_norm = pt.affPointTnf(theta_aff_ensemble, warped_points_aff_aff_norm)

        warped_points_aff_aff = PointsToPixelCoords(warped_points_aff_aff_norm, source_im_size/2)

        # Tolerance term
        L_pck = batch['L_pck'].data

        # Compute PCK (tau=0.01)
        correct_points_aff_1, num_points = correct_keypoints(warped_points_aff_GT.data, warped_points_aff.data, L_pck, tau=0.01)
        total_correct_points_aff_1 += correct_points_aff_1

        correct_points_aff_aff_1, _ = correct_keypoints(warped_points_aff_GT.data, warped_points_aff_aff.data, L_pck, tau=0.01)
        total_correct_points_aff_aff_1 += correct_points_aff_aff_1

        total_points_1 += num_points

        # Compute PCK (tau=0.03)
        correct_points_aff_3, num_points = correct_keypoints(warped_points_aff_GT.data, warped_points_aff.data, L_pck, tau=0.03)
        total_correct_points_aff_3 += correct_points_aff_3

        correct_points_aff_aff_3, _ = correct_keypoints(warped_points_aff_GT.data, warped_points_aff_aff.data, L_pck, tau=0.03)
        total_correct_points_aff_aff_3 += correct_points_aff_aff_3

        total_points_3 += num_points

        # Compute PCK (tau=0.05)
        correct_points_aff_5, num_points = correct_keypoints(warped_points_aff_GT.data, warped_points_aff.data, L_pck, tau=0.05)
        total_correct_points_aff_5 += correct_points_aff_5

        correct_points_aff_aff_5, _ = correct_keypoints(warped_points_aff_GT.data, warped_points_aff_aff.data, L_pck, tau=0.05)
        total_correct_points_aff_aff_5 += correct_points_aff_aff_5

        total_points_5 += num_points

        # print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

    # Print results
    print('')
    print('# ================ PCK Results ================ #')
    print('# Tau\t\t\t\t\t| 0.01\t| 0.03\t| 0.05  #')
    print('# ----------------------|-------|-------|------ #')
    PCK_aff_1 = total_correct_points_aff_1.cpu().numpy() / total_points_1
    PCK_aff_3 = total_correct_points_aff_3.cpu().numpy() / total_points_3
    PCK_aff_5 = total_correct_points_aff_5.cpu().numpy() / total_points_5
    print('# PCK affine\t\t\t| %.3f\t| %.3f\t| %.3f' % (PCK_aff_1,PCK_aff_3,PCK_aff_5), '#')

    PCK_aff_aff_1 = total_correct_points_aff_aff_1.cpu().numpy() / total_points_1
    PCK_aff_aff_3 = total_correct_points_aff_aff_3.cpu().numpy() / total_points_3
    PCK_aff_aff_5 = total_correct_points_aff_aff_5.cpu().numpy() / total_points_5
    print('# PCK affine (2-times)\t| %.3f\t| %.3f\t| %.3f' % (PCK_aff_aff_1,PCK_aff_aff_3,PCK_aff_aff_5), '#')

    print('# ============================================= #')

