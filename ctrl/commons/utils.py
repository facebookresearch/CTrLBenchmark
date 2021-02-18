# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re

import plotly


def plotly_rgb_to_hex(rgb_colors):
    """
    Convert a list of RGB strings in the format used by plotly ("rgb(<R>,<G>,<B>") to a list of hexadecimal codes.
    :param rgb_colors: List of RGB integer strings in the format ["rgb(255,0,0)", "rgb(0,255,0)", ...]
    :return: List of corresponding hex code strings ["#ff0000", "#00ff00", ...]
    """
    # Get the R, G and B components for each string
    color_codes = plotly_rgb_values(rgb_colors)
    # Format each rgb code in hex
    colors = ['#{:x}{:x}{:x}'.format(*cc) for cc in color_codes]
    return colors


def hex_to_rgb(hex_colors,):
    """
    """

    res = []
    for color in hex_colors:
        res.append([int(color[i:i+2], 16) for i in (1, 3, 5)])
    return res

def rgba_to_pl(rgb_color, alpha=False):
    """
    """
    # res = []
    # for color in rgb_colors:
    #     if not alpha:
    #         color = color[:3]
    return '#{:x}{:x}{:x}'.format(*rgb_color)


def plotly_rgb_values(rgb_colors):
    rgb_values = []
    for color in rgb_colors:
        vals = re.findall("rgb\(([0-9]+)\,\s?([0-9]+)\,\s?([0-9]+)\)", color)[0]
        vals = [int(val) for val in vals]
        rgb_values.append(vals)
    return rgb_values


default_colors = plotly_rgb_to_hex(plotly.colors.DEFAULT_PLOTLY_COLORS)
rainbow_colors = plotly_rgb_values(map(lambda col: col[1], plotly.colors.PLOTLY_SCALES['Rainbow']))

__all__ = [default_colors, rainbow_colors]
