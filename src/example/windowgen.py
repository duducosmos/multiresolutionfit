#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
from multiresolutionfit import Multiresoutionfit
from numpy import zeros, array
from numpy.random import randint
import matplotlib.pyplot as plt


scene1 = randint(256, size=(500, 500))
scene2 = randint(256, size=(500, 500))
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
'''

obj = Multiresoutionfit(scene1, scene2, verbose=True)
MAXW = min(scene1.shape[0], scene1.shape[1])
k=0.1
#wins = range(1, MAXW + 1)
wins = None
ftot, fw, wins = obj.ft(k=k, wins=wins)
z = obj.zvalue(k=k, wins=wins, permutations=30)

print(f"\nWeighted fit: {ftot:.2f}\n")
print(f"z value {z:.2f}.")
plt.plot(wins, fw, marker='D')
plt.xticks(wins)
#plt.ylim(ymax=0.95, ymin=0.75)
plt.xlim(xmax=MAXW, xmin=1)
plt.grid(True)
plt.show()
