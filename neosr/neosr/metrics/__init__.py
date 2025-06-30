# neosr/metrics/__init__.py (Corrected Version 2)

from copy import deepcopy
from typing import Any

import torch
from neosr.utils import get_root_logger
from neosr.utils.registry import METRIC_REGISTRY

# Import the actual calculation functions that we will call manually for cached models
from .calculate import calculate_dists as calculate_dists_func
from .calculate import calculate_topiq as calculate_topiq_func

_metric_models_cache = {}

def calculate_metric(data: dict, opt: dict[str, Any]) -> float:
    """Calculates a metric.

    This function now manages a cache for deep-learning-based metric models
    to prevent memory leaks during repeated validation.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop("type")

    # Handle DISTS metric
    if metric_type == 'calculate_dists':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'dists' not in _metric_models_cache:
            logger = get_root_logger()
            logger.info("Initializing and caching metric model: [DISTS] for the first time.")
            from neosr.losses.dists_loss import dists_loss
            model = dists_loss(as_loss=False).to(device)
            model.eval()
            _metric_models_cache['dists'] = model
        metric_result = calculate_dists_func(model=_metric_models_cache['dists'], **data, **opt)

    # Handle TOPIQ metric
    elif metric_type == 'calculate_topiq':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'topiq' not in _metric_models_cache:
            logger = get_root_logger()
            logger.info("Initializing and caching metric model: [TOPIQ] for the first time.")
            from neosr.metrics.topiq import topiq
            model = topiq().to(device)
            model.eval()
            _metric_models_cache['topiq'] = model
        metric_result = calculate_topiq_func(model=_metric_models_cache['topiq'], **data, **opt)

    else:
        # For simple metrics like PSNR/SSIM, just use the registry as before
        metric_result = METRIC_REGISTRY.get(metric_type)(**data, **opt)

    return metric_result

# Expose the public functions
__all__ = ["calculate_metric"]