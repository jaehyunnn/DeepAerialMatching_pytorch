import torch
from torchvision import transforms

class NormalizeImageDict(object):
    """
    
    Normalizes Tensor images in dictionary
    
    Args:
        image_keys (list): dict. keys of the images to be normalized
        normalizeRange (bool): if True the image is divided by 255.0s
    
    """
    
    def __init__(self,image_keys,normalizeRange=True):
        self.image_keys = image_keys
        self.normalizeRange=normalizeRange
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        for key in self.image_keys:
            if self.normalizeRange:
                sample[key] /= 255.0                
            sample[key] = self.normalize(sample[key])
        return  sample
    
    
def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalize or denormalize image tensor with ImageNet statistics."""
    im_size = image.size()
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=image.dtype, device=image.device).view(1, 3, 1, 1)

    if len(im_size) == 3:
        mean = mean.squeeze(0)
        std = std.squeeze(0)

    if forward:
        result = (image - mean) / std
    else:
        result = image * std + mean

    return result