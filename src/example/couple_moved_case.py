#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
import time
from multiresolutionfit import Multiresoutionfit
from numpy import zeros, array
import matplotlib.pyplot as plt
import cv2



scene1 = cv2.imread('./images/couple.png', 0).astype(int)
scenes2 = {"moved": cv2.imread('./images/couple_moved.png', 0).astype(int),
           "rotated": cv2.imread('./images/couple_rotated.png', 0).astype(int)}

MAXW = min(scene1.shape[0], scene1.shape[1])

for key in scenes2:
    print(f"Starting for {key}")
    obj = Multiresoutionfit(scene1, scenes2[key], verbose=True)
    k = 0.1
    ftot_par = obj.ft_par(k=k)
    print(f"\nWeighted fit par: {ftot_par:.2f}\n")
    ftot, fw, wins = obj.ft(k=k)
    print(f"\nWeighted fit: {ftot:.2f}\n")
    z = obj.zvalue(k=k)
    print(f"z value {z:.2f}.")
    plt.plot(wins, fw, marker='D')
    plt.xticks(wins)
    plt.grid(True)
    plt.show()
