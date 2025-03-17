custom_imports = dict(imports=[
    'DiagnosticFISH_package.DiagnosticFISH.src.backbones',
    'DiagnosticFISH_package.DiagnosticFISH.src.model', 
    'DiagnosticFISH_package.DiagnosticFISH.src.transforms', 
    'DiagnosticFISH_package.DiagnosticFISH.src.dataset', 
    ], allow_failed_imports=False)

default_scope = 'mmselfsup'

# dataset settings
h5_file = "/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/DiagnosticFISH/training_dataset.h5"

crop_sizes = [128]
num_views = [4]

view_pipelines = [
    [
        dict(type='C_RandomAffine', angle=(0, 360), scale=(1, 1), shift=(0,0), order=1),
        # dict(type='RandomIntensity', clip=True, low=1/3, high=3/1),
        # dict(type='RandomNoise', clip=True, mean=(0,0), std=(0,0.4)),
        dict(type='RandomFlip', prob=0.5),
        # dict(type='RandomGradient', clip=True, low=(0.0, 0.3), high=(0.7, 1.0)),
        # dict(type='RandomBlur', clip=True, kernel_range=(3, 3), sigma_range=(0, 1)),
        dict(type='CentralCutter', size=crop_size),
    ] for crop_size in crop_sizes
]

pipeline = [
    dict(type='MultiView', num_views=num_views, transforms=view_pipelines),
    dict(type='PackSelfSupInputs', meta_keys=["n_signals", 'size_nucleus'])
]


batch_size = 256
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    drop_last=True,
    dataset=dict(
        type='SingleChannelDataset',
        h5_file=h5_file,
        pipeline=pipeline,
        shuffle=False,
        )
    )

# model settings
# model = dict(
#     type='MVSimCLR',
#     num_views=4,
#     supervised_contrastive=True,
#     lam=0.1,
#     data_preprocessor=None,
#     backbone=dict(
#         type='SignalEncoder',
#         in_channels=1,
#         arctype='ConvNeXt_tiny',
#         classifier=None#dict(input_size=256, num_classes=16)
#         ),
#     neck=dict(
#         type='NonLinearNeck',  # SimCLR non-linear neck
#         in_channels=256,
#         hid_channels=256,
#         out_channels=64, 
#         num_layers=2,
#         with_avg_pool=True),
#     head=dict(
#         type='ContrastiveHead',
#         loss=dict(type='mmcls.CrossEntropyLoss'),
#         temperature=0.1),        
# )

model = dict(
    type='MVSimCLR',
    num_views=4,
    supervised_contrastive=True,
    lam=0.1,
    data_preprocessor=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,  # Adjust based on your input data (1 for grayscale, 3 for RGB)
        out_indices=(4,),
        norm_cfg=dict(type='BN'),
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth')  # MMPreTrain ResNet-50
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,  # ResNet-50 final output channels
        hid_channels=256,
        out_channels=64,
        num_layers=2,
        with_avg_pool=True
    ),
    head=dict(
        type='ContrastiveHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.05
    ),        
)

# optimizer
optimizer = dict(type='Adam', lr=0.0001, weight_decay=1e-4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={})
    )

# learning rate scheduler
n_iters = 2000
n_linear = 50
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=n_linear),
    dict(
        type='CosineAnnealingLR', 
        T_max=n_iters - n_linear, 
        by_epoch=False, 
        begin=n_linear, 
        end=n_iters)
]

# # eval_pipeline = [
# #     dict(type='C_TensorCombiner'),
# #     dict(type='CentralCutter', size=crop_sizes[0]),
# #     dict(type='PackSelfSupInputs')
# # ]

# # eval_batch_size = 128
# # eval_dataloader = dict(
# #     batch_size=eval_batch_size,
# #     num_workers=16,
# #     sampler=dict(type='DefaultSampler', shuffle=True),
# #     collate_fn=dict(type='default_collate'),
# #     drop_last=True,
# #     dataset=dict(
# #         patch_size=32,
# #         type=dataset_type,
# #         h5_file=h5_file,
# #         pipeline=eval_pipeline,
# #         append_central_mask=False,
# #         append_surrounding_masks=False,
# #         channels_to_skip=['DAPI-IF', 'GD2-IF', 'CD56-IF', 'MPO', 'H3K9Ac', 'H4K12Ac', 'CXCR2', 'IDO', 'clPARP', 'PNMT'],
# #         channels_to_use=None,
# #         exclude_edges=True,
# #         key='PROCESSED',
# #         shuffle=False,
# #         sample_types=['BM'],
# #         norm_raw=False,
# #         name2index=name2index
# #         )
# #     )

# # # using classification accuracy evaluation metric
# # val_evaluator = dict(type='RankMe', epsilon=1E-7)

# # val_cfg = dict(type='ValLoop')

# runtime settings
train_cfg = dict(
    type='IterBasedTrainLoop', 
    max_iters=n_iters)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=10, max_keep_ckpts=3),
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', log_metric_by_epoch=False, interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# #custom_hooks = [dict(type='RankMe', start=50, n_samples=50_000, eval_dataloader=eval_dataloader, epsilon=1E-100)]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(
    window_size=1,
    custom_cfg=[dict(data_src='', method='mean', window_size='global')])

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')

log_level = 'INFO'
load_from = None
resume = False