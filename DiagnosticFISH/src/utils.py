import re
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
from datetime import datetime
from typing import Union, List
from mmengine.config import Config
from mmselfsup.apis import init_model
from torch.utils.data import DataLoader
from mmselfsup.datasets.transforms import PackSelfSupInputs
from mmengine.dataset import default_collate, DefaultSampler
from DiagnosticFISH_package.DiagnosticFISH.src.transforms import CentralCutter, C_TensorCombiner
from DiagnosticFISH_package.DiagnosticFISH.src import dataset as custom_dataset


base_path = str(Path(__file__).resolve().parent.parent.parent)
print(base_path)

# Check if the path is already in PYTHONPATH
if base_path not in sys.path:
    print(f"Adding {base_path} to PYTHONPATH.")
    sys.path.append(base_path)
else:
    print(f"{base_path} is already in PYTHONPATH.")

def get_model(config_file: Path, checkpoint_file: Path, device):
    
    print(f"Loading Model from: {checkpoint_file}")
    
    model = init_model(
        config=str(config_file),
        checkpoint=checkpoint_file,
        device="cpu"
    )
    
    model.eval().to(device)
    
    return model

def get_model_dataloader(
    work_dir: Union[Path, str],
    dateformat: str='%Y%m%d_%H%M%S',
    meta_keys: List=[],
    batch_size: int=64,
    num_worker: int=16,
    patch_size: Union[int, None]=None,
    device: str="cuda",
    differing_config_file=None,
    shuffle=True):
    
    device = torch.device(device)
    
    checkpoint_file, config_file, log_file = get_work_dir_info(work_dir, dateformat)
    print(config_file)

    model = get_model(config_file=config_file, checkpoint_file=checkpoint_file, device=device)

    if differing_config_file is not None:
        dataset, _ = dataset_from_config(config_file=differing_config_file, meta_keys=meta_keys, patch_size=patch_size)
    else:
        dataset, _ = dataset_from_config(config_file=config_file, meta_keys=meta_keys, patch_size=patch_size)

    dataloader = DataLoader(
        dataset=dataset, 
        sampler=DefaultSampler(dataset, shuffle=shuffle), 
        batch_size=batch_size, 
        collate_fn=default_collate,
        num_workers=num_worker
    )
    
    return model, dataloader, dataset


def get_work_dir_info(work_dir, dateformat):
    
    work_dir = Path(work_dir)
    
    work_dir_content = list(Path(work_dir).glob("*"))
    latest_folder = get_latest(work_dir_content, dateformat)
    
    log_file = next(latest_folder.glob("*.log"))
    config_file = next(latest_folder.glob("*/config.py"))

    with open(next(work_dir.glob("last_checkpoint")), 'r') as f:
        checkpoint_file = f.readline()
    
    return checkpoint_file, config_file, log_file


def get_latest(folders, dateformat):
    
    newest = dict(folder=None, date= datetime.strptime("19700101_000000", dateformat))
    
    for folder in folders:
        if not folder.is_dir():
            continue
        
        curr_date = datetime.strptime(folder.stem, dateformat)
        if curr_date > newest["date"]:
            newest["folder"] = folder 
            newest["date"] = curr_date
        
    return newest["folder"]

BASE_KEYS = ['img', 'idx', 'size_nucleus', 'n_signals']

def dataset_from_config(config_file: Path, meta_keys: list, patch_size=None):
    
    add_keys_from_metadata = list(set(meta_keys) - set(BASE_KEYS))
    
    config = Config.fromfile(config_file)
        
    cutter_size = get_cutter_size(config_file)
    if patch_size is None:
        patch_size = cutter_size if cutter_size!=0 else dataset_kwargs["patch_size"]
    
    dataset_kwargs = config["train_dataloader"]["dataset"]
    dataset_kwargs["pipeline"] = [
        C_TensorCombiner(),
        CentralCutter(size=patch_size), 
        PackSelfSupInputs(meta_keys=meta_keys)
    ]
    
    dataset_kwargs['added_keys'] = add_keys_from_metadata
    
    dataset_type = dataset_kwargs.pop("type")
    
    dataset = getattr(custom_dataset, dataset_type)(
        **dataset_kwargs
    )

    return dataset, patch_size


def get_cutter_size(config_file):
    
    with open(config_file, "r") as f:
        lines = f.readlines()

    cutter_size = 0
    for line in lines:
        if "centralcutter" in line.lower() and "size" in line.lower():
            print(re.search(r'size=(\d+)', line))
            cutter_size = max(int(re.search(r'size=(\d+)', line).group(1)), cutter_size)

    return cutter_size
    
    
def run_model(model, dataloader, n_samples, get_images=False, mode="training", meta_keys=[]):
    
    results_dict = dict(embeddings=[])
    
    if meta_keys is not None:
        for mk in meta_keys:
            results_dict[mk] = []

    if get_images:
        images = []
        
    with torch.no_grad():

        for X in tqdm(dataloader, total=np.nanmin([n_samples//dataloader.batch_size, len(dataloader)]).astype(int)):
            
            inputs, metadata = model.data_preprocessor(X, True)

            if mode=="training":
                if hasattr(model.backbone, 'classifier'):
                    embeddings, predictions = model.extract_feat(inputs, mode="training", data_samples=metadata) # 
                else:
                    embeddings = model.extract_feat(inputs, data_samples=metadata) # 
                    predictions = [None]
                    
                embeddings = embeddings[0].detach().cpu().numpy()
                if predictions[0] is not None:
                    predictions = predictions[0].detach().cpu().numpy()
                results_dict["embeddings"].extend(embeddings)
            else:
                raise ValueError(f"{mode} is not a known mode!")
            
            for mk in meta_keys:
                results_dict[mk].extend([getattr(md, mk) for md in metadata])
            
            if get_images:
                images.extend(inputs[0].cpu().detach().numpy())

            if len(results_dict["embeddings"]) >= n_samples:
                break
        
        if get_images:
            results_dict["images"] = images
        
    return pd.DataFrame({key: sublist[:int(n_samples)] for key, sublist in results_dict.items()})
