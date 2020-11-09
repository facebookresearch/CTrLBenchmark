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

aircraft_taxonomy = ['707-320', '727-200', '737-200', '737-300', '737-400',
                     '737-500', '737-600', '737-700', '737-800', '737-900',
                     '747-100', '747-200', '747-300', '747-400', '757-200',
                     '757-300', '767-200', '767-300', '767-400', '777-200',
                     '777-300', 'A300B4', 'A310', 'A318', 'A319', 'A320',
                     'A321', 'A330-200', 'A330-300', 'A340-200', 'A340-300',
                     'A340-500', 'A340-600', 'A380', 'ATR-42', 'ATR-72',
                     'An-12', 'BAE 146-200', 'BAE 146-300', 'BAE-125',
                     'Beechcraft 1900', 'Boeing 717', 'C-130', 'C-47',
                     'CRJ-200', 'CRJ-700', 'CRJ-900', 'Cessna 172',
                     'Cessna 208', 'Cessna 525', 'Cessna 560',
                     'Challenger 600', 'DC-10', 'DC-3', 'DC-6', 'DC-8',
                     'DC-9-30', 'DH-82', 'DHC-1', 'DHC-6', 'DHC-8-100',
                     'DHC-8-300', 'DR-400', 'Dornier 328', 'E-170', 'E-190',
                     'E-195', 'EMB-120', 'ERJ 135', 'ERJ 145',
                     'Embraer Legacy 600', 'Eurofighter Typhoon', 'F-16A/B',
                     'F/A-18', 'Falcon 2000', 'Falcon 900', 'Fokker 100',
                     'Fokker 50', 'Fokker 70', 'Global Express',
                     'Gulfstream IV', 'Gulfstream V', 'Hawk T1', 'Il-76',
                     'L-1011', 'MD-11', 'MD-80', 'MD-87', 'MD-90',
                     'Metroliner', 'Model B200', 'PA-28', 'SR-20', 'Saab 2000',
                     'Saab 340', 'Spitfire', 'Tornado', 'Tu-134', 'Tu-154',
                     'Yak-42']

vdd_taxonomy = {
    'aircraft': [],
    'cifar100': [],
    'daimlerpedcls': [],
    'dtd': [],
    'gtsrb': [],
    'omniglot': [],
    'svhn': [],
    'ucf101': [],
    'vgg-flowers': [],
}

TAXONOMY = {'cifar100': cifar100_taxonomy,
            'cifar10': cifar10_taxonomy,
            'mnist': mnist_taxonomy,
            'svhn': mnist_taxonomy,
            'fashion-mnist': famnist_taxonomy,
            'dtd': dtd_taxonomy,
            'vdd': vdd_taxonomy,
            'aircraft': aircraft_taxonomy}


