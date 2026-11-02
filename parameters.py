from dataclasses import dataclass


@dataclass
class SSLParams:
    # dataset
    root_data: str = './data'
    name_data: str = 'colorMNIST' # colorMNIST, birds (Waterbirds), fairface, synthetic_engraving
    root_csv: str = './output.csv'
    experiment: str | None = None
    # Identifier used only to separate log/checkpoint directories.  It must not
    # alter the dataset experiment (e.g. CelebA's T=a/S=m specification).
    run_tag: str = ''
    intersectional_grouping: str = 'all'
    min_group_size: int = 2
    pretrained_backbone: bool = True
    backbone: str = 'dinov3_convnext'  # synthetic engraving: dinov3_convnext, vit_s8

    # optimization
    lr: float = 5e-2
    weight_decay: float = 5e-4
    epochs: int = 200
    batch_size: int = 256
    num_workers: int = 8
    devices: str | int = "auto"
    # SSL
    proj_out_dim: int = 256 # projection head output dimension (for colorMNIST its fix to 32)
    nb_global_views: int = 2 # fix also for every dataset because we're using cnn

    # loss
    loss_name: str = "uow" # uow, uow_intersectional, ow, infonce, lejepa, endunsupervised, fscl+, fscl*, crossentropy/ce
    reg_name: str = "mmd"
    kernel_name: str = "heat" # heat, sobolev, susreginduced    
    temperature: float = 5/256 #5/d (d=256) or 3/d (d=32)
    ell: int = 2 # for sobolev kernel
    # Weight assigned to the regularizer relative to invariance; must be in [0, 1].
    lambda_reg: float = 0.5
    # Deprecated explicit LeJEPA override. New runs use lambda_reg.
    lambda_lejepa: float | None = None
    # Deprecated explicit EnD override. New runs use lambda_reg.
    lambda_end: float | None = None
    # Fraction of the UOW regularization budget assigned to OWReg; must be in [0, 1].
    alpha: float | None = None
    # Deprecated alias for alpha, retained so existing launch commands continue to work.
    lambda_uow_reg: float | None = None
    nb_signs: int = 256 #only for engraving
    seed: int = 0

    # probing
    online_linear_probing: bool = True

    # dataloader
    accelerator: str = "auto"
    devices: str | int = "auto"
