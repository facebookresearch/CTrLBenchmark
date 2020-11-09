# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from ctrl.tasks.strategies.task_creation_strategy import TaskCreationStrategy
from ctrl.transformations.IdentityTransformation import \
    load_or_convert_to_image, crop_if_not_square
from ctrl.transformations.utils import BatchedTransformation
from torchvision.transforms import transforms


class InputDomainMutationStrategy(TaskCreationStrategy):
    def __init__(self, min_edit, max_edit, with_replacement, trans_trajectory,
                 *args, **kwargs):
        super(InputDomainMutationStrategy, self).__init__(*args, **kwargs)
        self.min_edit = min_edit
        self.max_edit = max_edit
        self.with_replacement = with_replacement
        self.trans_trajectory = trans_trajectory
        self.idx = 0

    def new_task(self, task_spec, concepts, trans, previous_tasks):
        cur_task_id = self.idx
        if self.trans_trajectory is not None:
            cur_trans_id = self.trans_trajectory[cur_task_id]
            first_usage = self.trans_trajectory.index(cur_trans_id)
            if first_usage < cur_task_id:
                allowed_trans = [previous_tasks[first_usage].transformation]
                exclude = None
            else:
                allowed_trans = None
                exclude = [t.transformation for t in previous_tasks if
                           hasattr(t.transformation, 'path')]
        else:
            exclude = None if self.with_replacement else \
                [t.transformation for t in previous_tasks]
            allowed_trans = None
        if self.trans_trajectory and cur_trans_id == None:
            trans = transforms.Compose([
                load_or_convert_to_image,
                # transforms.ToPILImage(),
                crop_if_not_square,
                transforms.ToTensor()
            ])
            new_transfo = BatchedTransformation(trans, 'Identity')
        elif self.min_edit < 0 or self.max_edit < 0:
            new_transfo = trans.get_transformation(exclude_trans=exclude,
                                                   allowed_trans=allowed_trans)
        else:
            new_transfo = trans.edit_transformation(task_spec.transformation,
                                                    self.min_edit,
                                                    self.max_edit)
        task_spec.transformation = new_transfo
        self.idx += 1
        return task_spec
