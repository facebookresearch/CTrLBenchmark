# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import collections
import numpy as np
from time import sleep
import torch
from datetime import datetime
from ignite.handlers import ModelCheckpoint


def fix_seed(seed):
    import numpy as np
    import random
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_basedir(root, add_timestamp=False, attempts=1):
    """Takes attempts shots at creating a folder from root,
    adding timestamp if desired.
    """
    for i in range(attempts):
        basedir = root
        if add_timestamp:
            timestamp = datetime.now().strftime("%a-%b-%d-%H:%M:%S.%f")
            basedir = os.path.join(basedir, timestamp)
        try:
            os.makedirs(basedir)
            return basedir
        except FileExistsError:
            sleep(1e-8)
    raise FileExistsError(root)


def init_checkpoint_handler(dirname, filename_prefix='', metric_name=None, higher_is_better=True,
                            score_function=None, score_name=None, **kwargs):
    
    if metric_name is not None:
        assert(score_function is None)
        assert(score_name is None)
        
        def metric_to_score_func(engine, metric_name):
            metrics = engine.state.metrics
            return metrics.get(metric_name, -np.inf)
            
        score_function = lambda engine: metric_to_score_func(engine, metric_name)
        score_name = metric_name

    if score_function is not None:
        final_score_function = lambda engine: -score_function(engine)
    else:
        final_score_function = score_function

    return ModelCheckpoint(
        dirname, filename_prefix,
        score_function=final_score_function,
        score_name=score_name,
        **kwargs
    )
