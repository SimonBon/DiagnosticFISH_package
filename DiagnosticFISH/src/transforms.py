import torchvision.transforms as T
from torch import nn
import numpy as np
from scipy.ndimage import zoom, shift, rotate

#import cv2

#from torchvision.transforms import v2

from mmselfsup.registry import TRANSFORMS
from mmcv.transforms.base import BaseTransform

from typing import Union, List, Tuple
from numbers import Number

ToTensor = T.ToTensor

@TRANSFORMS.register_module()
class C_RandomAffine(BaseTransform):
    
    def __init__(self, angle=(0,360), scale=(0.9, 1.1), shift=(-0.1,0.1), order=0):
        super().__init__()
        
        self.angle = angle
        self.scale = scale
        self.shift = shift
        self.order = order

        assert self.angle[0] <= self.angle[1], f'angle[0]: {angle[0]} must be smaller or equal to angle[1]: {angle[1]}'
        assert self.scale[0] <= self.scale[1], f'scale[0]: {scale[0]} must be smaller or equal to scale[1]: {scale[1]}'
        assert self.shift[0] <= self.shift[1], f'shift[0]: {shift[0]} must be smaller or equal to shift[1]: {shift[1]}'

    #@Timer(name='C_RandomAffine', text='Function '{name}' took {seconds:.6f} seconds to execute.')        
    def transform(self, results: dict) -> dict:
        '''Randomly crop the image and resize the image to the target size.

        Args:
            results (dict): Result dict from previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        height, width = img.shape[:2]

        # Random scaling
        curr_scale = np.random.uniform(self.scale[0], self.scale[1])

        # Random translation
        shift_y = np.random.uniform(self.shift[0], self.shift[1])
        shift_x = np.random.uniform(self.shift[0], self.shift[1])

        # Random rotation
        curr_angle = np.random.uniform(self.angle[0], self.angle[1])

        # Compute the combined transformation matrix
        center = (width / 2, height / 2)

        # Scaling matrix
        scale_matrix = cv2.getRotationMatrix2D(center, 0, curr_scale)

        # Translation matrix
        translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, curr_angle, 1)

        # Combine the transformation matrices
        transform_matrix = scale_matrix
        transform_matrix[0, 2] += translation_matrix[0, 2]
        transform_matrix[1, 2] += translation_matrix[1, 2]

        # Apply the combined transformation matrix
        img = cv2.warpAffine(img, transform_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Apply the rotation after scaling and translation
        img = cv2.warpAffine(img, rotation_matrix, (width, height), flags=self.order, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        
        if img.ndim == 2:
            img = np.expand_dims(img, -1)

        results['img'] = img
        results['angle'] = curr_angle
        results['scale'] = curr_scale
        results['shift'] = (shift_x, shift_y)
        
        return results
    
    
@TRANSFORMS.register_module()   
class CentralCutter(BaseTransform):
    
    def __init__(self, size: int):
        super().__init__()
        
        assert (size%2) == 0
        self.hsz = size // 2

    #@Timer(name='CentralCutter', text='Function '{name}' took {seconds:.6f} seconds to execute.')    
    def transform(self, results: dict) -> dict:
        '''Add random noise to the image with a mean and std deviation chosen randomly within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        
        c = img.shape[0] // 2, img.shape[1]//2
        
        #cut out the central part
        cropped_img = img[c[0]-self.hsz:c[0]+self.hsz, c[1]-self.hsz:c[1]+self.hsz]
        
        results['img'] = cropped_img
        results['cut_size'] = 2*self.hsz
        
        return results


@TRANSFORMS.register_module()   
class RandomNoise(BaseTransform):
    
    def __init__(self, mean=(0, 0), std=(0, 0.07), clip=True):
        super().__init__()
        
        self.mean = mean
        assert self.mean[0] <= self.mean[1]
        self.std = std
        assert self.std[0] <= self.std[1]
        
        self.clip = clip

    #@Timer(name='RandomNoise', text='Function '{name}' took {seconds:.6f} seconds to execute.')    
    def transform(self, results: dict) -> dict:
        '''Add random noise to the image with a mean and std deviation chosen randomly within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        
        # Randomly choose mean and std within the given range
        curr_mean = np.random.uniform(self.mean[0], self.mean[1])
        curr_std = np.random.uniform(self.std[0], self.std[1])
        
        # Add Gaussian noise to the image
        noise = np.random.normal(curr_mean, curr_std, img.shape)
        img = img + noise
        
        # Clip the values to be in the valid range
        if self.clip:
            if img.dtype == np.uint8:
                img = np.clip(img, 0, 255)
            else:  # assuming float type
                img = np.clip(img, 0.0, 1.0)
        
        results['img'] = img
        results['noise_level'] = (curr_mean, curr_std)
        
        return results
    
    
@TRANSFORMS.register_module()
class RandomIntensity(BaseTransform):
    
    def __init__(self, low: Union[List[Number], Number]=0.9, high: Union[List[Number], Number]=1.1, clip: bool=True):
        super().__init__()
        self.low = low
        self.high = high
        if isinstance(self.low, (list, tuple)) and isinstance(self.high, (list, tuple)):
            assert len(self.low) == len(self.high), 'low and high must be of same length or a float'
            assert all([l <= h for l,h in zip(self.low, self.high)]), f'all values of low must be lower than the corresponding value of high! {self.low} - {self.high}'
            
            self.create_list = False
            
        if isinstance(self.low, Number) and isinstance(self.high, Number):
            assert self.low  <= self.high
            
            self.create_list = True
            
        self.clip = clip


    #@Timer(name='RandomIntensity', text='Function '{name}' took {seconds:.6f} seconds to execute.')                
    def transform(self, results: dict) -> dict:
        '''Randomly adjust the intensity of the image channels within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        n_channels = img.shape[-1]
        
        # add functionality that if l and h is just a scalar value that it checks how man chanels are in an image and just copies l and h that often
        if self.create_list:
            curr_low = [self.low for _ in range(n_channels)]
            curr_high = [self.high for _ in range(n_channels)]
            
        else:
            curr_low = self.low
            curr_high = self.high

        # Randomly choose scaling factors for each channel within the given range
        channel_scaling = np.array([np.random.uniform(l, h) for l, h in zip(curr_low, curr_high)])
        
        # Apply scaling to each channel
        img = img * channel_scaling.reshape(1, 1, n_channels)
        
        # Clip the values to be in the valid range
        if self.clip:
            if img.dtype == np.uint8:
                img = np.clip(img, 0, 255)
            else:  # assuming float type
                img = np.clip(img, 0.0, 1.0)
        
        results['img'] = img
        results['channel_scaling'] = channel_scaling  # Store the scaling factors used
        
        return results


@TRANSFORMS.register_module() 
class C_ToTensor(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.tt = ToTensor()
        
    def forward(self, input_dict):
        return {k: self.tt(v) for k, v in input_dict.items()}
    
    
@TRANSFORMS.register_module()
class C_TensorCombiner(BaseTransform):
    
    def __init__(self):
        super().__init__()

    def transform(self, results) -> dict:
        """Concatenate image and mask tensors along the last dimension.

        Args:
            results (dict): Result dictionary containing 'img' and 'masks'.

        Returns:
            dict: Updated result dictionary with concatenated image and mask tensor.
        """
        if results['masks']:
            img = results['img']
            masks = np.atleast_3d(np.array(results['masks'])).transpose(1,2,0)  
            concat_tensor = np.concatenate((img, masks), axis=-1)
            results['img'] = concat_tensor
        
        return results
    
    
@TRANSFORMS.register_module()
class RandomGradient(BaseTransform):
    
    def __init__(self, low: Tuple[Number]=(0, 0), high: Tuple[Number]=(1, 1), clip: bool=True):
        super().__init__()
        
        self.low = low
        self.high = high 
        self.clip = clip


    #@Timer(name='RandomIntensity', text='Function '{name}' took {seconds:.6f} seconds to execute.')                
    def transform(self, results: dict) -> dict:
        '''Randomly adjust the intensity of the image channels within the specified range.

        Args:
            results (dict): Result dict from the previous pipeline.

        Returns:
            dict: Result dict with the transformed image.
        '''
        
        img = results['img']
        
        if img.ndim != 2:
            n_channels = img.shape[-1]
        else:
            n_channels = 1
        
        # add functionality that if l and h is just a scalar value that it checks how man chanels are in an image and just copies l and h that often
        
        gradient_images = []
        for _ in range(n_channels):
            
            curr_low = np.random.uniform(self.low[0], self.low[1])
            curr_high = np.random.uniform(self.high[0], self.high[1])

            theta = np.random.rand() * 2 * np.pi
            direction = np.array([np.cos(theta), np.sin(theta)])
                
            XX, YY = np.meshgrid(np.linspace(-1, 1 , img.shape[0]), np.linspace(-1, 1, img.shape[1]))
            
            directed_image = XX * direction[0] + YY * direction[1]
            directed_image = (directed_image - directed_image.min()) / (directed_image.max() - directed_image.min())
            scaled_directed_image = (curr_high - curr_low) * directed_image + curr_low
            
            gradient_images.append(scaled_directed_image)

        gradient_images = np.stack(gradient_images).transpose(1,2,0)
        
        results['img'] = np.clip(gradient_images + img, 0, 1) if self.clip else gradient_images + img
        
        return results       
    
    
@TRANSFORMS.register_module()
class RandomBlur(BaseTransform):

    def __init__(self, kernel_range: Tuple[int, int] = (3, 7), sigma_range: Tuple[float, float] = (0.1, 2.0), clip: bool = True):
        """
        Apply random Gaussian blur to the image.
        
        Args:
            kernel_range (Tuple[int, int]): Range for selecting the random kernel size. Must be odd.
            sigma_range (Tuple[float, float]): Range for selecting the random standard deviation for Gaussian blur.
            clip (bool): Whether to clip pixel values to [0,1] after transformation.
        """
        super().__init__()
        
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
        self.clip = clip

    def transform(self, results: dict) -> dict:
        """
        Apply random Gaussian blur to the image.

        Args:
            results (dict): Result dictionary containing 'img' key with the image data.

        Returns:
            dict: Updated results dictionary with blurred image.
        """
        img = results['img']

        if img.ndim != 2:
            n_channels = img.shape[-1]
        else:
            n_channels = 1

        blurred_images = []
        for c in range(n_channels):
            kernel_size = np.random.randint(self.kernel_range[0], self.kernel_range[1] + 1)
            if kernel_size % 2 == 0:  # Ensure kernel size is always odd
                kernel_size += 1
                
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])

            blurred_channel = cv2.GaussianBlur(img[..., c], (kernel_size, kernel_size), sigma)
            blurred_images.append(blurred_channel)

        blurred_images = np.stack(blurred_images, axis=-1) if n_channels > 1 else blurred_images[0]

        results['img'] = np.clip(blurred_images, 0, 1) if self.clip else blurred_images
        
        return results