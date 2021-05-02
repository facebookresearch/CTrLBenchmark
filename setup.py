# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import setuptools

setuptools.setup(
    name="ctrl-benchmark",
    version="0.0.3",
    author="Tom Veniat, Ludovic Denoyer & Marc'Aurelio Ranzato",
    license="MIT License",
    description="Continual Transfer Learning Benchmark",
    packages=setuptools.find_packages(),
    install_requires=[
        'pyyaml',
        'torch>=1.3,<2',
        'torchvision<1',
        'networkx>2,<3',
        'plotly',
        'pydot',
        'tqdm',
        'sklearn'
    ],
    include_package_data=True,
)