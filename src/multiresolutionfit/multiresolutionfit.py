#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Multiple Resolution Goodness of Fit.

The thesis of [cuevs2013]_ is that:

    "... there is no one 'proper' resolution, but rather a range of
         resolutions is necessary to adequately describe the fit of
         models with reality."

References
----------

.. [costanza89] COSTANZA, Robert. Model goodness of fit: a multiple resolution
                 procedure. Ecological modelling, v. 47, n. 3-4,
                 p. 199-215, 1989.

"""
from numpy import unique, asarray

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

        self.scene1 = scene1
        self.scene2 = scene2

        self.lines, self.cols = self.scene1.shape
        self.max_window_size = min(self.lines // 2, self.cols //  2)

    def count_class(self, window1, window2):
        unq1, cnts1 = unique(window1, return_coounts=True)
        unq2, cnts2 = unique(window2, return_coounts=True)

        cnt1 = asarray(unq1, cnts1)
        cnt2 = asarray(unq2, cnts2)
