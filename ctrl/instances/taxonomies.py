# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cifar100_taxonomy = {
    'aquatic mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruit and vegetables': ['apple', 'mushroom', 'orange', 'pear',
                             'sweet_pepper'],
    'household electrical devices': ['clock', 'keyboard', 'lamp', 'telephone',
                                     'television'],
    'household furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large man-made outdoor things': ['bridge', 'castle', 'house', 'road',
                                      'skyscraper'],
    'large natural outdoor scenes': ['cloud', 'forest', 'mountain', 'plain',
                                     'sea'],
    'large omnivores and herbivores': ['camel', 'cattle', 'chimpanzee',
                                       'elephant', 'kangaroo'],
    'medium-sized mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non-insect invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people': ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree',
              'willow_tree'],
    'vehicles 1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles 2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

cifar10_taxonomy = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog',
                    'frog', 'horse', 'ship', 'truck']

mnist_taxonomy = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                  '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

famnist_taxonomy = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
           'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

dtd_taxonomy = ['banded', 'blotchy', 'braided', 'bubbly', 'bumpy', 'chequered',
                'cobwebbed', 'cracked', 'crosshatched', 'crystalline',
                'dotted', 'fibrous', 'flecked', 'freckled', 'frilly', 'gauzy',
                'grid', 'grooved', 'honeycombed', 'interlaced', 'knitted',
                'lacelike', 'lined', 'marbled', 'matted', 'meshed', 'paisley',
                'perforated', 'pitted', 'pleated', 'polka-dotted', 'porous',
                'potholed', 'scaly', 'smeared', 'spiralled', 'sprinkled',
                'stained', 'stratified', 'striped', 'studded', 'swirly',
                'veined', 'waffled', 'woven', 'wrinkled', 'zigzagged']

TAXONOMY = {'cifar100': cifar100_taxonomy,
            'cifar10': cifar10_taxonomy,
            'mnist': mnist_taxonomy,
            'svhn': mnist_taxonomy,
            'fashion-mnist': famnist_taxonomy,
            'dtd': dtd_taxonomy,
            }

