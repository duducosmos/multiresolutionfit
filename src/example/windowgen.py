#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
from multiresolutionfit import Multiresoutionfit
from numpy import zeros, array
from numpy.random import randint
import matplotlib.pyplot as plt

'''
scene1 = randint(256, size=(50, 50))
scene2 = randint(256, size=(50, 50))
'''

scene1 = array([[1, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
                [1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
                [3, 3, 2, 2, 3, 3, 3, 3, 3, 3],
                [1, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [1, 1, 1, 3, 3, 3, 3, 3, 3, 3],
                [2, 2, 2, 2, 2, 2, 2, 2, 3, 3],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [3, 3, 3, 3, 2, 2, 3, 3, 3, 3],
                [3, 3, 3, 3, 2, 2, 2, 2, 3, 3]
                ])

scene2 = array([[1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
                [1, 1, 1, 1, 2, 3, 3, 3, 3, 3],
                [1, 1, 1, 2, 3, 3, 3, 3, 3, 3],
                [3, 1, 2, 2, 3, 3, 3, 4, 4, 4],
                [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [1, 1, 1, 3, 3, 3, 3, 3, 3, 3],
                [1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
                [1, 2, 2, 3, 3, 2, 2, 3, 3, 3],
                [3, 3, 3, 3, 2, 2, 2, 3, 3, 3],
                [3, 3, 3, 3, 2, 2, 2, 2, 3, 3]
                ])


obj = Multiresoutionfit(scene1, scene2)
wins = [1]
MAXW = min(scene1.shape[0], scene1.shape[1])
w = 2
while w <= MAXW:
    wins.append(w)
    w *= 2
wins.append(MAXW)



#wins = obj.window_generator()
wins = array(wins)
fw, ftot = obj.ft(wins, k=0.1)
print(ftot)

plt.plot(wins, fw, marker='D')
plt.xticks(wins)
#plt.ylim(ymax=0.95, ymin=0.75)
#plt.xlim(xmax=10, xmin=1)
plt.grid(True)
plt.show()
