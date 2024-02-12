import yaml
import logging
import os
from typing import Generator, Any

# import timm
# import segmentation_models_pytorch as smp
import torch
# from monai.losses import TverskyLoss
# from pytorch_toolbelt.losses import (DiceLoss, SoftBCEWithLogitsLoss,
#                                      BinaryFocalLoss)
# from sklearn.metrics import roc_auc_score

# from src.utils.losses import BCEDiceLoss, FocalDiceLoss


def get_config(config_path: str) -> dict:
    """Load config file from give path

    Parameters
    ----------
    config_path : str
        Path to config file

    Returns
    -------
    dict
        dict-like config.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_logger() -> logging.Logger:
    """Create and setup logger. Always in DEBUG mode.

    Returns
    -------
    logging.Logger
        Logger object.
    """

    #  setup logger
    logging.getLogger('cmai').handlers.clear()
    logger = logging.getLogger('cmai')
    logger_handler = logging.StreamHandler()
    logger_handler.setLevel(logging.DEBUG)
    logger_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s: %(message)s'))
    logger.addHandler(logger_handler)
    logger.setLevel(logging.DEBUG)

    return logger


def make_result_dir(prefix: str = '') -> str:
    """Make directory to save all results.

    Parameters
    ----------
    prefix : str
        Directory prefix part before number part
    n_folds : int
        Number of fords for cross-validation training.
        Every fold has own result directory.

    Returns
    -------
    str
        Path to result dir
    """

    res_dir_root = os.path.join('./results', prefix)
    os.makedirs(res_dir_root, exist_ok=True)

#     res_dirs = os.listdir(res_dir_root)
#     res_dirs = list(filter(lambda x: x.isdigit(), res_dirs))
#     if res_dirs:
#         res_dir = max(map(int, res_dirs)) + 1
#     else:
#         res_dir = 1
#     res_dir = '{:0>4}'.format(res_dir)
#     res_dir = os.path.join(res_dir_root, res_dir)
    return res_dir_root


def get_model(**config: Any) -> torch.nn.Module:
    """Get model according to given config.

    Returns
    -------
    torch.nn.Module
        Model built according to given config.
    """
    mode = config.get('mode', 'seg')
    if mode == 'seg':
        models = {
            'unet': smp.Unet,
            'fpn': smp.FPN,
        }
        return models[config['architecture']](**config['params'])

    elif mode == 'cls':
        return timm.create_model(config['architecture'], **config['params'])


def get_activation(**config: dict) -> torch.nn.Module:
    """Get activation function according to given config.

    Returns
    -------
    torch.nn.Module
        Activation module built according to given config.
    """

    activations = {
        'sigmoid': torch.nn.Sigmoid,
        'softmax': torch.nn.Softmax  #  dim=1
    }
    return activations[config['name']](**config['params'])


def get_loss(**config: dict) -> torch.nn.Module:
    """Get loss function according to given config.

    Returns
    -------
    torch.nn.Module
        Loss module built according to given config.
    """

    losses = {
        # 'binary_dice_focal_loss': DiceFocalLossBinary,
        'dice_loss': DiceLoss,
        'bce': SoftBCEWithLogitsLoss,
        'bce_dice': BCEDiceLoss,
        'focal': BinaryFocalLoss,
        'tverskiy_loss': TverskyLoss,
        'focal_dice_loss': FocalDiceLoss,

        #  classification
        'ce': torch.nn.CrossEntropyLoss,
        'bce': torch.nn.BCEWithLogitsLoss
    }
    return losses[config['name']](**config['params'])


def get_optimizer(parameters: Generator,
                  **config: dict) -> torch.optim.Optimizer:
    """Get optimizer according to given config.

    Parameters
    ----------
    parameters : Generator
        Model parameters to be optimized (learnt). Usually: `model.parameters()`.

    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer
    """

    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }
    return optimizers[config['name']](parameters, **config['params'])


def get_metric(**config: Any):
    """Get metric function according to given config.

    Returns
    -------
    torch.nn.Module
        Metric module built according to given config.
    """

    metrics = {
        'cls': {
            'roc_auc': roc_auc_score
        },
        'seg': {
            'iou': smp.utils.metrics.IoU
        }
    }

    mode = config.get('mode', 'seg')

    if mode == 'seg':
        return metrics[mode][config['name']](**config['params'])
    elif mode == 'cls':
        return metrics[mode][config['name']]
    else:
        return None


def get_lr_scheduler(optimizer, **config) -> object:
    """Get learning rate scheduler according to given config.

    Returns
    -------
    torch.nn.Module
        LR scheduler built according to given config.
    """

    lr_schedulers = {
        'rlrop': torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    return lr_schedulers[config['name']](optimizer, **config['params'])

