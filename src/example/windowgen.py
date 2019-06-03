#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
from multiresolutionfit import Multiresoutionfit
from numpy import zeros, array, dot
from numpy.random import randint
import matplotlib.pyplot as plt
import cv2


scene1 = cv2.imread('./images/couple.png', 0).astype(int)
scene2 = cv2.imread('./images/couple13.png', 0).astype(int)

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
k = 0.1
#wins = range(1, MAXW + 1)
wins = None
ftot_par = obj.ft_par(k=k, wins=wins, npixels=100)
print(f"\nWeighted fit par: {ftot_par:.2f}\n")
ftot, fw, wins = obj.ft(k=k, wins=wins)
print(f"\nWeighted fit: {ftot:.2f}\n")
plt.plot(wins, fw, marker='D')
plt.xticks(wins)
#plt.ylim(ymax=0.95, ymin=0.75)
plt.xlim(xmax=MAXW, xmin=1)
plt.grid(True)
plt.show()
'''
z = obj.zvalue(k=k, wins=wins, permutations=30)
print(f"z value {z:.2f}.")
'''
