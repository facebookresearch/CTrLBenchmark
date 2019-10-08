# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from src.modules.Independent_moredata_ll import IndependentMDLLModel
from src.modules.Independent_ub_ll import IndependentUBLLModel
from src.modules.change_layer_llmodel import ChangeLayerLLModel
from src.modules.experience_replay_llmodel import ExperienceReplayLLModel
from src.modules.fine_tune_head_llmodel import FineTuneHeadLLModel
from src.modules.fine_tune_leg_llmodel import FineTuneLegLLModel
from src.modules.independent_llmodel import IndependentLLModel
from src.modules.multitask_head_llmodel import MultitaskHeadLLModel
from src.modules.multitask_leg_llmodel import MultitaskLegLLModel
from src.modules.ewc_llmodel import EWCLLModel
from src.modules.new_head_llmodel import NewHeadLLModel
from src.modules.new_leg_llmodel import NewLegLLModel
from src.modules.same_model_llmodel import SameModelLLModel
from . import _utils


def get_module_by_name(name):
    if name == 'change-layer':
        return ChangeLayerLLModel
    if name == 'independent':
        return IndependentLLModel
    if name == 'independent-upper-bound':
        return IndependentUBLLModel
    if name == 'independent-more-data':
        return IndependentMDLLModel
    if name == 'new-head':
        return NewHeadLLModel
    if name == 'new-leg':
        return NewLegLLModel
    if name == 'multitask-head':
        return MultitaskHeadLLModel
    if name == 'multitask-leg':
        return MultitaskLegLLModel
    if name == 'same-model':
        return SameModelLLModel
    if name == 'finetune-head':
        return FineTuneHeadLLModel
    if name == 'finetune-leg':
        return FineTuneLegLLModel
    if name == 'ewc':
        return EWCLLModel
    if name == 'er':
        return ExperienceReplayLLModel
    if name.endswith('-dict'):
        return OrderedDict
    raise NotImplementedError(name)


def init_module(**kwargs):
    for k, v in kwargs.items():
        if isinstance(v, dict):
                v = init_module(**v)
        kwargs[k] = v
    if '_name' in kwargs:
        return get_module_by_name(kwargs.pop('_name'))(**kwargs)
    else:
        return kwargs

