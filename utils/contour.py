import numpy as np
import scipy.ndimage as ndimage


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

    return np.sum(contour, axis=-1), np.sum(edt, axis=-1)

