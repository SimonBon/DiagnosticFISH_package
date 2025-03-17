# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import einops 

import numpy as np

from mmselfsup.registry import MODELS
from mmselfsup.models.builder import BACKBONES
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer
from mmselfsup.models.algorithms.base import BaseModel

from . import backbones

@MODELS.register_module()
class MVSimCLR(BaseModel):

    def __init__(self, num_views: int=2, supervised_contrastive=False, lam=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.num_views = num_views
        self.supervised_contrastive = supervised_contrastive
        self.lam = lam
        
    # @staticmethod
    # def _create_buffer(batch_size: int, num_views: int, device: torch.device, n_signals=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #     N = batch_size * num_views # how many samples are there in total
    #     mask = 1 - torch.eye(N, dtype=torch.uint8).to(device) # create ones matrix with zeroed diagonal

    #     pos_idx = (
    #         torch.arange(N).to(device),
    #         num_views * torch.arange(batch_size, dtype=torch.long).unsqueeze(1).repeat(1, num_views).view(-1, 1).squeeze().to(device)
    #     )
        
    #     print(pos_idx)
        
    #     neg_mask = torch.ones((N, N - 1), dtype=torch.uint8).to(device)
    #     neg_mask[pos_idx] = 0
        
    #     return mask, pos_idx, neg_mask
    
    @staticmethod
    def _create_buffer(batch_size: int, num_views: int, device: torch.device, n_signals=None) -> Tuple[torch.Tensor, torch.Tensor]:

        N = batch_size * num_views # how many samples are there in total

        # Create signal_mask initialized with an identity matrix
        if n_signals is not None:
            signal_mask = np.eye(batch_size, dtype=np.uint8)

            # Convert n_signals to a NumPy array
            n_signals = np.array(n_signals)

            # Find unique signals with their counts
            unique, counts = np.unique(n_signals, return_counts=True)

            # Identify values that appear at least twice
            repeated_values = unique[counts >= 2]

            # Update signal_mask for repeated values
            for value in repeated_values:
                idxs = np.where(n_signals == value)[0]
                signal_mask[np.ix_(idxs, idxs)] = 1
                
        else:
            signal_mask = np.eye(batch_size, dtype=np.uint8)
  
        pos_mask = np.kron(signal_mask, np.ones((num_views, num_views)))
        pos_mask[np.eye(N) == 1] = 0

        pos_idx = np.where(pos_mask)
        
        neg_mask = np.ones((N, N))
        neg_mask[pos_idx] = 0

        return torch.tensor(pos_mask, dtype=torch.bool).to(device), torch.tensor(neg_mask, dtype=torch.bool).to(device)

    def extract_feat(self, inputs: List[torch.Tensor], data_samples=None, **kwargs) -> Tuple[torch.Tensor]:
        return self.backbone(inputs[0], **kwargs)
    
    
    def loss(self, inputs: List[torch.Tensor], data_samples: List[SelfSupDataSample], **kwargs) -> Dict[str, torch.Tensor]:
        
        assert isinstance(inputs, list)
        assert len(inputs) == self.num_views, f'Expected {self.num_views} views, got {len(inputs)}'

        n_signals = torch.tensor([ds.n_signals for ds in data_samples])
        batch_size = inputs[0].size(0)
        
        latents, predictions = [], []
        for i in range(self.num_views):

            x = inputs[i]
            if hasattr(self.backbone, 'classifier'):
                x, preds = self.backbone(x)
            else:
                x = self.backbone(x)
                preds = [None]
            z = self.neck(x)[0]  # (batch_size)x(d)
            z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
            latents.append(z)
            predictions.append(preds[0])
            
        latents = torch.stack(latents)
        z = einops.rearrange(latents, 'views batch_size dimensions -> (batch_size views) dimensions', batch_size=batch_size, views=self.num_views)
        z = torch.cat(GatherLayer.apply(z), dim=0)  # (num_views * batch_size * num_gpus)x(d)
        N_total = z.size(0) // self.num_views

        similarity = torch.matmul(z, z.permute(1, 0))  # (num_views * N_total)x(num_views * N_total)
                
        if self.supervised_contrastive:
            pos_mask, neg_mask = self._create_buffer(N_total, self.num_views, similarity.device, n_signals)
        else:
            pos_mask, neg_mask = self._create_buffer(N_total, self.num_views, similarity.device)
        
    # Extract positive and negative similarities

        pos_similarity = similarity * pos_mask.float()
        neg_similarity = similarity * neg_mask.float()
        
        pos_sim_avg = (torch.sum(pos_similarity, axis=1) / torch.sum(pos_mask, axis=1)).unsqueeze(1)
        neg_sim_avg = (torch.sum(neg_similarity, axis=1) / torch.sum(neg_mask, axis=1)).unsqueeze(1)
        
        # Compute supervised contrastive loss
        loss = self.head(pos_sim_avg, neg_sim_avg)
        
        print(predictions[0])
        if predictions[0] is not None:
            predictions = torch.stack(predictions)
            predictions = einops.rearrange(predictions, 'views batch_size dimensions -> (batch_size views) dimensions', batch_size=batch_size, views=self.num_views)
            prediction_loss = self.prediction_loss(predictions, n_signals)
        
            return dict(loss=loss + self.lam * prediction_loss, pred_loss=prediction_loss)
        
        else:
            return dict(loss=loss)
    
    
    def prediction_loss(self, predictions, n_signals):
        """
        Compute cross-entropy loss for predictions.
        Args:
            predictions: Tensor of shape [batch_size * views, num_classes] (logits)
            n_signals: Tensor of shape [batch_size] (integer class labels)
        """
        # Ensure n_signals is not one-hot encoded but remains class indices
        n_signals = torch.stack([n_signals] * self.num_views)
        n_signals = einops.rearrange(n_signals, 'views batch_size -> (batch_size views)').to(predictions.device)

        # Compute cross-entropy loss (expects class indices, not one-hot)
        loss = F.cross_entropy(predictions, n_signals)

        return loss

@BACKBONES.register_module()
class SignalEncoder(nn.Module):

    def __init__(self, in_channels, arctype, backbone_kwargs={}, classifier=None):
        super().__init__()

        self.in_channels = in_channels
        self.backbone = getattr(backbones, arctype)(in_channels, **backbone_kwargs)
        self.classifier = Classifier(**classifier) if classifier is not None else None

    def forward(self, input, mode="training"):

        x = self.backbone(input)
        
        out_features = F.normalize(x)
        

        predictions = self.classifier(out_features.squeeze()) if self.classifier is not None else None
        
        assert mode in ["training"]
        
        if mode == "training":
            return tuple([out_features]), tuple([predictions])

    def init_weights(self, pretrained=None):
        return

    def train(self, mode=True):
        super().train(mode)
        

from .utils import get_model, get_work_dir_info
from copy import deepcopy
from mmselfsup.models.utils.data_preprocessor import SelfSupDataPreprocessor


class FISHEncoder(nn.Module):
    
    def __init__(self, work_dir):
        super().__init__()
        
        self.data_preprocessor = SelfSupDataPreprocessor()
        
        dateformat = '%Y%m%d_%H%M%S'
        
        checkpoint_file, config_file, _ = get_work_dir_info(work_dir, dateformat)

        self.red_encoder = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device='cpu')
        self.green_encoder = deepcopy(self.red_encoder)

    def forward(self, RGB_FISH):

        assert RGB_FISH.ndim == 4
        
        red_channel = [RGB_FISH[:, 0].unsqueeze(1)]
        green_channel = [RGB_FISH[:, 1].unsqueeze(1)]
        nuc_channel = RGB_FISH[:, 2]
        
        red_features = self.red_encoder(red_channel)[0].squeeze()
        green_features = self.green_encoder(green_channel)[0].squeeze()
        
        out_features = torch.cat((red_features, green_features), dim=1)
        
        return tuple([out_features])
    
    def extract_feat(self, inputs: List[torch.Tensor], data_samples=None, **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Backbone outputs.
        
        """   
        
        device = next(self.parameters()).device
        
        return self.forward(inputs[0].to(device))
        

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(p=0.25)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x