#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
"""
Example from [costanza89]_.

References
----------

.. [costanza89] COSTANZA, Robert. Model goodness of fit: a multiple resolution
                 procedure. Ecological modelling, v. 47, n. 3-4,
                 p. 199-215, 1989.
"""
from multiresolutionfit import Multiresoutionfit
from numpy import  arange, array
from numpy.random import randint
import matplotlib.pyplot as plt

size = (400, 400)
k = 0.1

scene1 = randint(256, size=size)
scene2 = randint(256, size=size)

obj = Multiresoutionfit(scene1, scene2, verbose=True)
ftot, fw, wins = obj.ft(k=k)
print(f"\nWeighted fit: {ftot:.2f}\n")
z, ftot_par = obj.zvalue(k=k, wins=wins)
print(f"\nWeighted fit par: {ftot_par:.2f}\n")
print(f"z value {z:.2f}.")
plt.plot(wins, fw, marker='D')
plt.title("Random Data")
plt.xticks(wins)
plt.grid(True)
plt.show()
