"""Dataset class"""

import numpy as np
import os
import glob
from torch.utils.data import Dataset
from .preprocessing import transform_pair


class UnpairedDataset(Dataset):
    """Unpaired dataset class for images of dataset A and B.
    """
    
    def __init__(self, root, path_a, path_b, size=256, convert_rgb=False, contours=False, augment=False, corrupt=False):
        """Initialise dataset
        
        Args:
            root (string): root path containing folders to the data
            path_a (string): path to dataset A, this is compulsory.
            path_b (string): path to dataset B, leave this for single dataset.
            size (int): size of image for the NN
            convert_rgb (bool): flag to convert image to 3 / RGB image
        """
        self.files_a = list(glob.glob(os.path.join(root, path_a) + '/*.*'))
        
        if path_a != None and path_b != None:
            self.single_dataset = False
            self.files_b = list(glob.glob(os.path.join(root, path_b) + '/*.*'))
            self.files = list(zip(self.files_a, self.files_b))
        else:
            self.single_dataset = True
            self.files = list(zip(self.files_a))
        
        self.augment = augment
        self.size = size
        self.convert_rgb = convert_rgb
        self.contours = contours
        # only for domain A!
        self.corrupt = corrupt
        

    def __getitem__(self, index):
        """Get an unpaired pair of images from dataset A and B and
        corresponding mask and sobel edges.
        
        Args:
            index (int): index of dataset array
        
        Returns
            { A: item_a, B: item_b } (dict): actual return value
            
            item_a (np.array, np.array, np.array): three tuples containing
            transformed image, mask and sobel edges
            
            item_b (np.array, np.array, np.array): same as item_a but 
            from dataset B
        """
        pair = {}

        item_a = np.load(self.files[index][0])
        res_a = transform_pair(item_a['image'], 
                               item_a['mask'],
                               augment=self.augment,
                               convert_rgb=self.convert_rgb, 
                               contours=self.contours, 
                               corrupt=self.corrupt)        
        pair['A'] = res_a

        if self.single_dataset == False:
            item_b = np.load(self.files[index][1])
            res_b = transform_pair(item_b['image'], 
                                   item_b['mask'], 
                                   augment=self.augment,
                                   convert_rgb=self.convert_rgb, 
                                   contours=self.contours)
            pair['B'] = res_b
            
        return pair
    
    def __len__(self):
        """Returns the shorter length of the two datasets"""
        if self.single_dataset:
            return len(self.files_a)
        return min(len(self.files_a), len(self.files_b))

    