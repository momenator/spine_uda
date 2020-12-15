"""Functions for processing data before passing them to the network
"""

import torch
import numpy as np
import os
import glob
from skimage.filters import sobel, gaussian
import scipy.ndimage as ndimage
from skimage.morphology import dilation, erosion


def get_seg_contour_dist_trans(img):
    """Get segmentation contour and distance transform given the segmentation mask.
    Taken from: https://github.com/ferchonavarro/shape_aware_segmentation/.
    
    Args:
        img (np.array): image containing segmentation mask
    
    Returns:
        contour (np.array): segmentation contour
        edt (np.array): segmentation distance transform
    """
    
    img_one_hot = np.eye(100)[np.uint8(img)] > 0.0

    contour = np.uint8(np.zeros(img_one_hot.shape))
    edt = np.zeros(img_one_hot.shape)

    for i in range(1, 100):
        if np.sum(img_one_hot[:, :, i]) != 0:
            # fill holes
            img = ndimage.morphology.binary_fill_holes(img_one_hot[:, :, i])

            # extract contour
            contour[:, :, i] = ndimage.morphology.binary_dilation(img == 0.0) & img

            # distance transform
            tmp = ndimage.morphology.distance_transform_edt(img)
            edt[:, :, i] = (tmp - np.amin(tmp)) / (np.amax(tmp) - np.amin(tmp))

    return np.sum(contour, axis=-1).astype(np.uint8), np.sum(edt, axis=-1).astype(np.float32)


def square_image(image, pad_value=0):
    """Pad image array image such the shorter dimension is the same
    as the longer one. Dimensions may be different (by 1 pixel) 
    if the longer dimension is odd.
    
    Args:
        image (np.array): 2D or 3D image array
        pad_value (int): padding constant
        
    Returns
        padded (np.array): padded M with equal height and width
    """
    
    if len(image.shape) == 2:
        (a,b) = image.shape
        if a > b:
            padding=((0,0),(((a-b)//2),((a-b)//2)))
        else:
            padding=((((b-a)//2),((b-a)//2)),(0,0))
    else:
        (n,a,b) = image.shape
        if a > b:
            padding=((0,0),(0,0),(((a-b)//2),((a-b)//2)))
        else:
            padding=((0,0),(((b-a)//2),((b-a)//2)),(0,0))
    
    return np.pad(image,padding,mode='constant',constant_values=pad_value)


def random_square_crop_pair(image, mask, size):
    """Randomly crop square region (size*size) image array 
    and corresponding mask.
    
    Args:
        image (np.array): 2D image array
        mask (np.array): corresponding 2D image mask
        size (int): length of crop on both height and width dimension
    
    Returns:
        cropped img (np.array): cropped img
        cropped mask (np.array): cropped mask
    
    """
    if len(image.shape) != 2:
        raise Exception('unknown dimension')

    h, w = image.shape

    assert((size <= h and size <= w) == True)

    x = np.random.randint(w - size)
    y = np.random.randint(h - size)
    
    return image[y:y+size,x:x+size], mask[y:y+size,x:x+size]


def smoothen_binary_mask(mask, threshold=0.4):
    """Smoothen the edges of binary mask by gaussian blur 
    and thresholding. The reason for this function is that the
    mask may not be binary-valued and thresholding the mask with
    non zero values to 1 will produce jagged and coarse edges.
    
    Args:
        mask (np.array): 2D mask array
        threshold (float): threshold value for smoothing
        
    Returns:
        smooth (np.array): smoothed mask
    
    """
    # ensure values are  [0, 1]
    divisor = (mask.max() - mask.min())
    smooth = mask
    if divisor != 0:
        smooth = (mask - mask.min()) / (mask.max() - mask.min())
    
    smooth = gaussian(smooth)
    smooth[smooth < threshold] = 0
    smooth[smooth >= threshold] = 1
    return smooth


def augment_pair(image, mask):
    """Apply image augmentation (random noise and flipping) on image
    
    Args:
        image (np.array): image to be augmented
        mask (np.array): corresponding mask
    
    Returns:
        augmented image (np.array): augmented image
        augmented mask (np.array): augmented mask
    """
    
    # apply augmentation here
    is_flip_h = np.random.random() 
    is_flip_v = np.random.random()
    is_gauss = np.random.random()
    mean = 0
    std = 0.0001

    if is_gauss > 0.5:
        image = image + np.random.normal(mean, std, image.shape)

    if is_flip_h > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)

    if is_flip_v > 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)

    
    return image, mask


def to_torch_tensor(np_array):
    """Convert 2D numpy array to 3D torch tensor (1 channel)
    
    Args:
        arr (np.array): arr to be converted
        
    Returns:
        arr (torch.Tensor): 3D array in torch.Tensor format
    """
    
    if len(np_array.shape) == 2:
        np_array = np_array.reshape((1, np_array.shape[0], np_array.shape[1]))
        
    # TODO - converting everything to float is not efficient!
    return torch.from_numpy(np_array).to(dtype=torch.float32)


def transform_pair(image, mask, size=256, sobel_edges=True, augment=False, convert_rgb=False, contours=False, corrupt=False):
    """Transform image and mask pair with random cropping and augmentation
    (random noise and flipping) so that it can be processed by NN.
    Provide sobel_edges by default.
    
    Args:
        image (np.array): 2D image array
        mask (np.array): 2D corresponding mask
        sobel_edges (bool): flag to indicate whether to include sobel edges
        convert_rgb (bool): flag to convert image to 3 / RGB image
    
    Returns:
        image (np.array): 3D image torch tensor
        mask (np.array): 3D mask torch tensor
        ?sobel (np.array)
    """
    
    # smoothen mask
    mask = smoothen_binary_mask(mask)
    
    if augment:
        image, mask = augment_pair(image, mask)

    image = square_image(image)
    mask = square_image(mask)
    
    image, mask = random_square_crop_pair(image, mask, size)
    sobel_image = sobel(image)
    corrupted_mask = corrupt_image(mask)

    if convert_rgb:
        image = np.array([image, image, image])
        sobel_image = np.array([sobel_image, sobel_image, sobel_image])
        
    if contours:
        ctr, edt = get_seg_contour_dist_trans(mask)
        ctr = to_torch_tensor(ctr)
        edt = to_torch_tensor(edt)

    image = to_torch_tensor(image)
    mask = to_torch_tensor(mask)
    sobel_image = to_torch_tensor(sobel_image)
    corrupted_mask = to_torch_tensor(corrupted_mask)
    
    results = [image, mask]
    
    if sobel_edges:
        results.append(sobel_image)
        
    if corrupt:
        results.append(corrupted_mask)
    
    if contours:
        results.append(ctr)
        results.append(edt)
    
    return results


def corrupt_image(img):
    """Corrupt image (greyscale) function 
    Args:
        img (np.array): Greyscale image to be corrupted
    
    Returns:
        img (np.array): Image to be corrupted
    """
    # function for corruption
    
    salt_noise = np.random.rand(*img.shape)
    salt_noise = np.where(salt_noise > 0.999, img.max(), 0)
    salt_noise = dilation(salt_noise, selem=np.ones((9, 9)))
    
    corrupt_mask = np.random.rand(*img.shape)
    corrupt_img = img * np.around(corrupt_mask)
    corrupt_img = dilation(corrupt_img)
    # increase kernel size for more corruption
    corrupt_img = erosion(corrupt_img, np.random.rand(3, 3))
    corrupt_img = np.where(corrupt_img != 0, corrupt_img, salt_noise)
    
    return corrupt_img

