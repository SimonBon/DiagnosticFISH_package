batch_size = 256
crop_sizes = [
    96,
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'DiagnosticFISH_package.DiagnosticFISH.src.backbones',
        'DiagnosticFISH_package.DiagnosticFISH.src.model',
        'DiagnosticFISH_package.DiagnosticFISH.src.transforms',
        'DiagnosticFISH_package.DiagnosticFISH.src.dataset',
    ])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=10, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=1, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    runtime_info=dict(type='RuntimeInfoHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'))
default_scope = 'mmselfsup'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
h5_file = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/DiagnosticFISH/training_dataset.h5'
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    custom_cfg=[
        dict(data_src='', method='mean', window_size='global'),
    ],
    window_size=1)
model = dict(
    backbone=dict(
        arctype='ConvNeXt_tiny', in_channels=1, type='SignalEncoder'),
    data_preprocessor=None,
    head=dict(
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.1,
        type='ContrastiveHead'),
    neck=dict(
        hid_channels=256,
        in_channels=256,
        num_layers=2,
        out_channels=64,
        type='NonLinearNeck',
        with_avg_pool=True),
    num_views=4,
    supervised_contrastive=True,
    type='MVSimCLR')
n_iters = 1000
n_linear = 50
num_views = [
    4,
]
optim_wrapper = dict(
    optimizer=dict(lr=1, momentum=0.9, type='LARS', weight_decay=1e-06),
    paramwise_cfg=dict(custom_keys=dict()),
    type='OptimWrapper')
optimizer = dict(lr=1, momentum=0.9, type='LARS', weight_decay=1e-06)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=50, start_factor=0.0001, type='LinearLR'),
    dict(
        T_max=950,
        begin=50,
        by_epoch=False,
        end=1000,
        type='CosineAnnealingLR'),
]
pipeline = [
    dict(
        num_views=[
            4,
        ],
        transforms=[
            [
                dict(clip=True, high=2, low=0.5, type='RandomIntensity'),
                dict(
                    clip=True,
                    mean=(
                        0,
                        0,
                    ),
                    std=(
                        0,
                        0.07,
                    ),
                    type='RandomNoise'),
                dict(
                    angle=(
                        0,
                        360,
                    ),
                    order=1,
                    scale=(
                        0.8,
                        1.25,
                    ),
                    shift=(
                        0,
                        0,
                    ),
                    type='C_RandomAffine'),
                dict(prob=0.5, type='RandomFlip'),
                dict(size=96, type='CentralCutter'),
            ],
        ],
        type='MultiView'),
    dict(meta_keys=[
        'n_signals',
        'size_nucleus',
    ], type='PackSelfSupInputs'),
]
resume = False
train_cfg = dict(max_iters=1000, type='IterBasedTrainLoop')
train_dataloader = dict(
    batch_size=256,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        h5_file=
        '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/data/DiagnosticFISH/training_dataset.h5',
        pipeline=[
            dict(
                num_views=[
                    4,
                ],
                transforms=[
                    [
                        dict(
                            clip=True, high=2, low=0.5,
                            type='RandomIntensity'),
                        dict(
                            clip=True,
                            mean=(
                                0,
                                0,
                            ),
                            std=(
                                0,
                                0.07,
                            ),
                            type='RandomNoise'),
                        dict(
                            angle=(
                                0,
                                360,
                            ),
                            order=1,
                            scale=(
                                0.8,
                                1.25,
                            ),
                            shift=(
                                0,
                                0,
                            ),
                            type='C_RandomAffine'),
                        dict(prob=0.5, type='RandomFlip'),
                        dict(size=96, type='CentralCutter'),
                    ],
                ],
                type='MultiView'),
            dict(
                meta_keys=[
                    'n_signals',
                    'size_nucleus',
                ],
                type='PackSelfSupInputs'),
        ],
        shuffle=False,
        type='SingleCellDataset'),
    drop_last=True,
    num_workers=16,
    sampler=dict(shuffle=True, type='DefaultSampler'))
view_pipelines = [
    [
        dict(clip=True, high=2, low=0.5, type='RandomIntensity'),
        dict(clip=True, mean=(
            0,
            0,
        ), std=(
            0,
            0.07,
        ), type='RandomNoise'),
        dict(
            angle=(
                0,
                360,
            ),
            order=1,
            scale=(
                0.8,
                1.25,
            ),
            shift=(
                0,
                0,
            ),
            type='C_RandomAffine'),
        dict(prob=0.5, type='RandomFlip'),
        dict(size=96, type='CentralCutter'),
    ],
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SelfSupVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/home/simon_g/isilon_images_mnt/10_MetaSystems/MetaSystemsData/_simon/src/DiagnosticFISH_package/configs/work_dirs/home/config'
