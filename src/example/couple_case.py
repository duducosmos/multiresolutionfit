#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
import time
from multiresolutionfit import Multiresoutionfit
from numpy import zeros, array
import matplotlib.pyplot as plt
import cv2



scene1 = cv2.imread('./images/couple.png', 0).astype(int)
scenes2 = {"7%": cv2.imread('./images/couple07.png', 0).astype(int),
           "13%": cv2.imread('./images/couple13.png', 0).astype(int),
           "25%": cv2.imread('./images/couple25.png', 0).astype(int),
           "50%": cv2.imread('./images/couple50.png', 0).astype(int),
           "75%": cv2.imread('./images/couple75.png', 0).astype(int),
           "88%": cv2.imread('./images/couple88.png', 0).astype(int)
          }

MAXW = min(scene1.shape[0], scene1.shape[1])

for key in scenes2:
    print(f"Starting for {key}")
    obj = Multiresoutionfit(scene1, scenes2[key], verbose=True)
    k = 0.1
    z, ftot_par = obj.zvalue(k=k, permutations=20, npixels=100)
    print(f"z value {z:.2f}.")
    print(f"\nWeighted fit par: {ftot_par:.2f}\n")
