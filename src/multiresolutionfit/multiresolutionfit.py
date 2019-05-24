#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Multiple Resolution Goodness of Fit.

The thesis of [cuevs2013]_ is that:

    "...there is no one `proper` resolution, but rather a range of
         resolutions is necessary to adequately describe the fit of
         models with reality."

References
----------

.. [costanza89] COSTANZA, Robert. Model goodness of fit: a multiple resolution
                 procedure. Ecological modelling, v. 47, n. 3-4,
                 p. 199-215, 1989.

"""
from numpy import unique, asarray, append
from boundbox2d import BoundBox2D
import matplotlib.pyplot as plt
from random import randint
from math import floor, ceil

class ImageSizeError(Exception):
    def __init__(self, value):
        value = value

    def __str__(self):
        return repr(value)


class Multiresoutionfit:
    r"""
    Multiple Resolution Goodness of Fit.

    :param 2d_array scene1: Gray scale image
    :param 2d_array scene2: Gray scale image
    """
    def __init__(self, scene1, scene2):

        if scene1.shape[0] != scene2.shape[0]:
            raise ImageSizeError("The images have diferent number of lines")

        if scene1.shape[1] != scene2.shape[1]:
            raise ImageSizeError("The images have diferent number of columns")

        self._scene1 = scene1
        self._scene2 = scene2

        self._lines, self._cols = self._scene1.shape
        self._golden_ratio = (1.0 + 5.0 ** 0.5) / 2.0

    def window_generator(self):
        cl = min(self._lines, self._cols)
        w =  floor(cl / self._golden_ratio)
        wins = []
        i = 1
        while w >= 1:
            wins.append(w)
            w =  floor(w / self._golden_ratio ** i)
            i += 1
        return wins

    def dif_class(self, win):
        for i in range(self._lines - win):
            for j in range(self._cols - win):
                self.count_class(self._scene1[i:i + win, j:j + win],
                                       self._scene1[i:i + win, j:j + win])
                print(i,j)


    def count_class(self, window1, window2):
        unq1, cnts1 = unique(window1, return_counts=True)
        unq2, cnts2 = unique(window2, return_counts=True)

        cnt1 = list(zip(unq1, cnts1))
        cnt2 = list(zip(unq2, cnts2))
        return cnt1, cnt2
