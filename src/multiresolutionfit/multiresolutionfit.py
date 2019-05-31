#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Multiple Resolution Goodness of Fit.

The thesis of [cuevs2013]_ is that:

    "...there is no one `proper` resolution, but rather a range of
         resolutions is necessary to adequately describe the fit of
         models with reality."

License
-------

Developed by: E. S. Pereira.
e-mail: pereira.somoza@gmail.com

Copyright [2019] [E. S. Pereira]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

References
----------

.. [costanza89] COSTANZA, Robert. Model goodness of fit: a multiple resolution
                 procedure. Ecological modelling, v. 47, n. 3-4,
                 p. 199-215, 1989.

"""
from math import floor, ceil
from numpy import array, exp
from .countclass import countclass


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

    def golden_rectangle_generator(self):
        r"""
        Golden rectangle Generator.

        Calculate the Golden Rectangle side that is possible to draw inside the
        image.
        :return int w: side of golden rectangles.
        """
        cl = min(self._lines, self._cols)
        w =  floor(cl / self._golden_ratio)
        wins = []
        i = 1
        while w >= 1:
            wins.append(w)
            w =  floor(w / self._golden_ratio ** i)
            yield w
            i += 1

    def _f(self, win, window1, window2):
        """
        Count class.
        Return $f = 1 - \frac{\sum_{i=1}^{p}{|a_{1i} -a_{2i}|}}{2w^{2}}$
        """
        cnt1, cnt2 = countclass(window1, window2)
        A = set(cnt1.keys())
        B = set(cnt2.keys())
        common = list(A.intersection(B))
        only_in_A = list(A - B)
        only_in_B = list(B - A)

        aki = 0

        for cl in common:
            aki += abs(cnt1[cl] - cnt2[cl])
        for cl in only_in_A:
            aki += cnt1[cl]
        for cl in only_in_B:
            aki += cnt2[cl]
        f = 1 - aki / (2.0 * win * win)
        return f


    def fwin(self, win):
        """
        Fit at a particular sampling window size.
        :parameter int win: window size
        Return:  $F_{w}= \frac{\sum_{s=1}^{t_{w}}{
                  \left[ 1 - \frac{\sum_{i=1}^{p}{|a_{1i} -a_{2i}|}}{2w^{2}}
                  \right]_{s}}}{t_{w}}$
        """
        fw = 0
        for i in range(self._lines - win):
            for j in range(self._cols - win):
                f = self._f(win, self._scene1[i:i + win, j:j + win],
                   self._scene2[i:i + win, j:j + win])
                fw += f

        n = ((self._lines - win) *  (self._cols - win))
        if n == 0:
            fw = self._f(win, self._scene1, self._scene2)
        else:
            fw = fw / n
        return fw

    def ft(self, k, wins=None):
        """
        Weight average of the fits over all window sizes.
        :parameter float k: weight range [0,1].
        :parameter list wins:  list of windows size.
        """
        fw = []
        if wins is None:
            wins = self.window_generator()
        for win in wins:
            fw.append(self.fwin(win))
        fw = array(fw)
        wins = array(wins)
        e = exp( - k * (wins - 1))
        ftot = (fw * e).sum() / e.sum()
        return fw, ftot
