from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FeatureExtraction(nn.Module):
    """Feature extraction using timm backbones with manual layer selection."""

    def __init__(self, train_fe=True, use_cuda=True, feature_extraction_cnn='vgg'):
        super(FeatureExtraction, self).__init__()

        if feature_extraction_cnn == 'vgg':
            full_model = timm.create_model('vgg16', pretrained=True)
            # Extract features up to pool4 (15x15 output for 240x240 input)
            self.model = nn.Sequential(*list(full_model.features.children())[:24])

        elif feature_extraction_cnn == 'resnet101':
            full_model = timm.create_model('resnet101', pretrained=True)
            self.model = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.act1,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3,
            )

        elif feature_extraction_cnn == 'resnext101':
            full_model = timm.create_model('resnext101_32x4d', pretrained=True)
            self.model = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.act1,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3,
            )

        elif feature_extraction_cnn == 'se_resnext101':
            full_model = timm.create_model('seresnext101_32x4d', pretrained=True)
            self.model = nn.Sequential(
                full_model.conv1,
                full_model.bn1,
                full_model.act1,
                full_model.maxpool,
                full_model.layer1,
                full_model.layer2,
                full_model.layer3,
            )

        elif feature_extraction_cnn == 'densenet169':
            full_model = timm.create_model('densenet169', pretrained=True)
            # Extract up to denseblock3 (indices 0-7, 15x15 output for 240x240 input)
            self.model = nn.Sequential(*list(full_model.features.children())[:8])

        else:
            raise ValueError(f"Unknown backbone: {feature_extraction_cnn}")

        if not train_fe:
            for param in self.model.parameters():
                param.requires_grad = False

        if use_cuda:
            self.model.cuda()

    def forward(self, image_batch):
        return self.model(image_batch)

class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        return F.normalize(feature, p=2, dim=1, eps=1e-6)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        return correlation_tensor


class FeatureRegression(nn.Module):
    def __init__(self, output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(15 * 15, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x

class AerialNetBase(nn.Module):
    """Base class for aerial image matching networks."""

    def __init__(self, geometric_model='affine', normalize_features=True,
                 normalize_matches=True, use_cuda=True,
                 feature_extraction_cnn='se_resnext101', train_fe=False):
        super(AerialNetBase, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = FeatureExtraction(
            train_fe=train_fe, use_cuda=use_cuda,
            feature_extraction_cnn=feature_extraction_cnn
        )
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        output_dim = 6 if geometric_model == 'affine' else 6
        self.FeatureRegression = FeatureRegression(output_dim, use_cuda=use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

    def extract_features(self, image):
        """Extract and optionally normalize features from image."""
        features = self.FeatureExtraction(image)
        if self.normalize_features:
            features = self.FeatureL2Norm(features)
        return features

    def compute_correlation(self, feature_A, feature_B):
        """Compute correlation between features and optionally normalize."""
        correlation = self.FeatureCorrelation(feature_A, feature_B)
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
        return correlation


class AerialNetSingleStream(AerialNetBase):
    """Single-stream network for aerial image matching.

    Uses bidirectional correlation to estimate affine transformation
    between source and target images.
    """

    def forward(self, tnf_batch):
        # Extract features
        feature_A = self.extract_features(tnf_batch['source_image'])
        feature_B = self.extract_features(tnf_batch['target_image'])

        # Compute bidirectional correlation
        correlation_AB = self.compute_correlation(feature_A, feature_B)
        correlation_BA = self.compute_correlation(feature_B, feature_A)

        # Regress transformation parameters
        theta_AB = self.FeatureRegression(correlation_AB)
        theta_BA = self.FeatureRegression(correlation_BA)

        return theta_AB, theta_BA


class AerialNetTwoStream(AerialNetBase):
    """Two-stream network with jittered target for robust matching.

    Extends single-stream by adding a jittered target stream for
    improved robustness during training.
    """

    def forward(self, tnf_batch):
        # Extract features from all images
        feature_A = self.extract_features(tnf_batch['source_image'])
        feature_B = self.extract_features(tnf_batch['target_image'])
        feature_C = self.extract_features(tnf_batch['target_image_jit'])

        # Compute bidirectional correlation for original pair
        correlation_AB = self.compute_correlation(feature_A, feature_B)
        correlation_BA = self.compute_correlation(feature_B, feature_A)

        # Compute bidirectional correlation for jittered pair
        correlation_AC = self.compute_correlation(feature_A, feature_C)
        correlation_CA = self.compute_correlation(feature_C, feature_A)

        # Regress transformation parameters
        theta_AB = self.FeatureRegression(correlation_AB)
        theta_BA = self.FeatureRegression(correlation_BA)
        theta_AC = self.FeatureRegression(correlation_AC)
        theta_CA = self.FeatureRegression(correlation_CA)

        return theta_AB, theta_BA, theta_AC, theta_CA