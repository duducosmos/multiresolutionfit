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
        self._max_window_size = min(self._lines // 2, self._cols //  2)

    def reposition(self, box, init_pos, nbox, wins):
        maxrecursion = 100
        for i in range(0, nbox):
            x = init_pos[i][0]
            y = init_pos[i][1]
            for j in range(i + 1, nbox):
                x2 = init_pos[j][0]
                y2 = init_pos[j][1]
                contrec = 0
                while box[i].collision(box[j]) is True:
                    box[i].vertices -= 1
                    box[j].vertices += 1

                    if (box[i].vertices >= self._cols).any() :
                        box[i].vertices -= 1

                    if (box[j].vertices >= self._cols).any() :
                        box[j].vertices -= 1

                    if (box[i].vertices >= self._lines).any() :
                        box[i].vertices -= 1

                    if (box[j].vertices >= self._lines).any() :
                        box[j].vertices -= 1

                    if (box[i].vertices < 0).any() :
                        box[i].vertices += 1

                    if (box[j].vertices < 0).any() :
                        box[j].vertices += 1

                    contrec += 1
                    if contrec == maxrecursion:
                        return True

        for i in range(0, nbox):
            for j in range(i + 1, nbox):
                if box[i].collision(box[j]) is True:
                    return True
        return False

    def _windgen(self, wins):
        nbox = len(wins)
        init_pos = [[randint(0, self._lines - wi - 1),
                     randint(0, self._cols - wi - 1)]
                     for wi in wins ]

        box = [BoundBox2D([[init_pos[i][0], init_pos[i][1]],
                           [init_pos[i][0] + wins[i], init_pos[i][1]],
                           [init_pos[i][0] + wins[i], init_pos[i][1] + wins[i]],
                           [init_pos[i][0], init_pos[i][1] + wins[i]]
                           ]
                         ) for i in range(nbox)
              ]

        return box, init_pos, nbox, wins


    def window_generator(self):
        wins = [self._max_window_size]
        w = self._max_window_size
        while w > 1:
            w //= 2
            wins.append(w)
        box, init_pos, nbox, wins = self._windgen(wins)
        print(nbox)

        teste = self.reposition(box, init_pos, nbox, wins)
        while teste is True:
            box, init_pos, nbox, wins = self._windgen(wins)
            teste = self.reposition(box, init_pos, nbox, wins)


        fig = plt.figure(1, figsize=(5, 5), dpi=90)
        ax = fig.add_subplot(111)
        for i in range(0, nbox):
            vtmp = box[i].vertices.copy()
            vtmp = append(vtmp, [vtmp[0, :]], axis=0)
            ax.plot(vtmp[:, 0], vtmp[:, 1], alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2,
                    label="{}".format(i))
        plt.show()



    def count_class(self, window1, window2):
        unq1, cnts1 = unique(window1, return_coounts=True)
        unq2, cnts2 = unique(window2, return_coounts=True)

        cnt1 = asarray(unq1, cnts1)
        cnt2 = asarray(unq2, cnts2)
