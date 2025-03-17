import os
import math
from typing import Union, List, Literal, Optional
from pathlib import Path

from warnings import warn
import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from mmcv.transforms import Compose

# from tqdm import tqdm

# import sys
# sys.path.append("/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/mmselfsup_package")
# sys.path.append("/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/PARIS_codebase")

# import PARIS
from mmselfsup.registry import DATASETS

import os
DEBUG = os.getenv('MMDEBUG', 'false').lower() == 'true'


import time
from functools import wraps

def timeit(debug=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not debug:
                return func(*args, **kwargs)
                
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Get the class name
            class_name = args[0].__class__.__name__ if args else 'N/A'
            
            print(f"{class_name}({func.__name__}): {execution_time:e}")
            
            return result
        
        return wrapper
    return decorator


@DATASETS.register_module()
class SingleChannelDataset(Dataset):

    def __init__(self,
                h5_file: Union[Path, str],
                shuffle: bool = False,
                pipeline: List[List[dict]] = None,
                channel_idx: int = None,
                masked_idxs: List = None,
                **kwargs):
        
        super().__init__()
        
        self.masked_idxs = masked_idxs
        
        # Ensure the HDF5 file exists
        assert Path(h5_file).exists(), f"Provided path to h5 file does not exist: {h5_file}"
        self.h5_file = h5py.File(h5_file, 'r')
        print(f"H5-File: {h5_file}")
        
        self.shuffle = shuffle
        self.channel_idx = channel_idx
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = lambda x: x

            
    def __len__(self):
        if self.masked_idxs is not None: 
            return len(self.masked_idxs)
            
        return len(self.h5_file["FISH"])

    @timeit(debug=DEBUG)
    def __getitem__(self, idx, ommit_pipeline=False):
        
        if self.masked_idxs is not None: 
            idx = self.masked_idxs[idx]
        
        if self.channel_idx is not None:
            signal_image = np.expand_dims(self.h5_file["FISH"][idx, ..., self.channel_idx], -1)
        else:
            signal_image = np.expand_dims(self.h5_file["FISH"][idx], -1)
        size = self.h5_file["NUCLEUS"][idx].sum()
        n_signals = self.h5_file["N_SIGNALS"][idx]

        pipeline_dict = {
            'img': signal_image,
            #'nucleus_img': nucleus_image,
            'idx': idx,
            'n_signals': n_signals,
            'size_nucleus': size,
            'masks': []
        }

        if ommit_pipeline:
            return pipeline_dict
                    
        return self.pipeline(pipeline_dict)
    
    
@DATASETS.register_module()
class RGBDataset(Dataset):

    def __init__(self,
                h5_file: Union[Path, str],
                shuffle: bool = False,
                pipeline: List[List[dict]] = None,
                **kwargs):
        
        super().__init__()
        
        # Ensure the HDF5 file exists
        assert Path(h5_file).exists(), f"Provided path to h5 file does not exist: {h5_file}"
        self.h5_file = h5py.File(h5_file, 'r')
        print(f"H5-File: {h5_file}")
        
        self.shuffle = shuffle
        
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        else:
            self.pipeline = lambda x: x

            
    def __len__(self):
        return len(self.h5_file["FISH"])

    @timeit(debug=DEBUG)
    def __getitem__(self, idx, ommit_pipeline=False):
        
        signal_image = self.h5_file["FISH"][idx]
        size = self.h5_file["NUCLEUS"][idx].sum()
        n_signals = self.h5_file["N_SIGNALS"][idx]

        pipeline_dict = {
            'img': signal_image,
            #'nucleus_img': nucleus_image,
            'idx': idx,
            'n_signals': n_signals,
            'size_nucleus': size,
            'masks': []
        }

        if ommit_pipeline:
            return pipeline_dict
                    
        return self.pipeline(pipeline_dict)