# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_results(x, nsample=1, res=1., renorm=True, cmap='viridis'):
    """
    x is a list of tuples (name, value) where value
    is a dict of with keys 'in' and 'out', corresponding
    to the input and output sequences given to the model 
    """
    # retrieving max image value to not renorm
    
    vmin = np.inf
    vmax = - np.inf
    for _, v in x:
        for l in v:
            if v[l][0][0].shape[0] != 2:
                vls = [vli[:nsample] for vli in v[l]]
                vmin = min(vmin, np.min(vls))
                vmax = max(vmax, np.max(vls))

    x = OrderedDict(x)
    # calculating max column length
    maxincols, maxoutcols = 0, 0
    for k in x:
        if 'in' in x[k]:
            maxincols = max(len(x[k]['in']), maxincols)
        if 'out' in x[k]:
            maxoutcols = max(len(x[k]['out']), maxoutcols)
    cols = maxincols + maxoutcols
    rows = len(x) * nsample

    def plot_one(im, title='', cmap='viridis', renorm=True):
        if im.shape[0] == 2:
            raise NotImplementedError
        plt.axis('off')
        plt.imshow(im.squeeze(), origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
    
    plt.figure(figsize=(cols * res, rows * res))
    for i, k in enumerate(x):
        for s in range(nsample):
            if 'in' in x[k]:
                
                title = '{}, in'.format(k)
                for t in range(len(x[k]['in'])):
                    n = s * len(x) * cols + cols * i + t
                    im = x[k]['in'][t][s]
                    plt.subplot(rows, cols, n + 1)
                    plot_one(im, title, cmap=cmap, renorm=renorm)
            if 'out' in x[k]:
                title = '{}, out'.format(k)
                for t in range(maxincols, maxincols + len(x[k]['out'])):
                    im = x[k]['out'][t - maxincols][s]
                    n = s * len(x) * cols + cols * i + t
                    plt.subplot(rows, cols, n + 1)
                    plot_one(im, title, cmap=cmap, renorm=renorm)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.001, hspace=0.1)
    return plt.gcf()


def from_matplotlib(fig, to_tensor=False):
    fig.canvas.draw()
    rgb = fig.canvas.tostring_rgb()
    plt.close(fig)
    data = np.fromstring(rgb, dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    if to_tensor:
        data = torch.from_numpy(data.transpose(2, 0, 1))

    return data



def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image
