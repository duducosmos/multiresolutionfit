#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
from multiresolutionfit import Multiresoutionfit
from numpy import zeros
from numpy.random import randint

scene1 = randint(256, size=(100, 56))
scene2 = randint(256, size=(100, 56))

obj = Multiresoutionfit(scene1, scene2)

obj.window_generator()
