# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from src.modules.Independent_moredata_ll import IndependentMDLLModel



class IndependentUBLLModel(IndependentMDLLModel):
    def __init__(self, train_samples, *args, **kwargs):
        super(IndependentUBLLModel, self).__init__(*args, **kwargs)
        self.train_samples = train_samples
        self.file_name_format = '{t_name}_{n_samples}_upper_bound'

    def get_shots(self, task):
        return [self.train_samples, *task.n_samples_per_class[1:]]

