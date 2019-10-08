# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from src.modules.change_layer_llmodel import ChangeLayerLLModel


class IndependentMDLLModel(ChangeLayerLLModel):
    def __init__(self, n_samples_max, *args, **kwargs):
        super(IndependentMDLLModel, self).__init__(*args, **kwargs)
        self.file_name_format = '{t_name}_{n_samples}_more_data'
        self.n_samples_max = n_samples_max

    def get_shots(self, task):
        train_samples = task.n_samples_per_class[0] * (task.id + 1)
        train_samples = min(train_samples, self.n_samples_max)
        return [train_samples, *task.n_samples_per_class[1:]]

    def prepare_task(self, task, training_params):
        # data_path = training_params.pop('data_path')
        concepts = task.src_concepts
        attrs = task.attributes
        transfo = task.transformation

        n_samples_per_class = self.get_shots(task)
        name = self.file_name_format.format(t_name=task.name,
                                            n_samples=n_samples_per_class[0])
        save_path = os.path.dirname(task.save_path[0])

        fake_task = task.generator._create_task(concepts, attrs, transfo,
                                                n_samples_per_class, name,
                                                save_path)
        training_params['data_path'] = fake_task.save_path
