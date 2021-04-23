"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
from depthmerge.data.base_dataset import BaseDataset, get_transform
from depthmerge.data.image_folder import make_dataset
from depthmerge.util.guidedfilter import GuidedFilter
from PIL import Image
import numpy as np

import torch

class TemplateDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        # parser.set_defaults(input_nc=1, output_nc=1,preprocess='none')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        self.dir_outer = opt.test_data_outer
        self.dir_inner = opt.test_data_inner

        self.outer_paths = sorted(make_dataset(self.dir_outer, opt.max_dataset_size))
        self.inner_paths = sorted(make_dataset(self.dir_inner, opt.max_dataset_size))

        self.dataset_size = len(self.inner_paths)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        normalize_coef = np.float32(2 ** (16))

        data_outer = Image.open(self.outer_paths[index % self.dataset_size])  # needs to be a tensor
        org_size = data_outer.size

        data_outer = data_outer.resize((672,672))
        data_outer = np.array(data_outer, dtype=np.float32)
        data_outer = data_outer / normalize_coef


        data_inner = Image.open(self.inner_paths[index % self.dataset_size])  # needs to be a tensor
        data_inner = data_inner.resize((672,672))
        data_inner = np.array(data_inner, dtype=np.float32)
        data_inner = data_inner / normalize_coef


        data_outer = torch.from_numpy(data_outer)
        data_outer = torch.unsqueeze(data_outer, 0)
        data_outer  = self.normalize(data_outer)

        data_inner = torch.from_numpy(data_inner)
        data_inner = torch.unsqueeze(data_inner,0)
        data_inner  = self.normalize(data_inner)

        image_path = self.outer_paths[index % self.dataset_size]
        return {'data_inner': data_inner, 'data_outer': data_outer, 'image_path': image_path, 'image_size':org_size}

    def __len__(self):
        """Return the total number of images."""
        return self.dataset_size

    def normalize(self,input):
        input = input * 2
        input = input - 1
        return input
