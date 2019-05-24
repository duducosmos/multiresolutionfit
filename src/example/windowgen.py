#!/usr/bin/env python3
# -*- Coding: UTF-8 -*-
from multiresolutionfit import Multiresoutionfit
from numpy import zeros
from numpy.random import randint

scene1 = randint(256, size=(500, 500))
scene2 = randint(256, size=(500, 500))

obj = Multiresoutionfit(scene1, scene2)

obj.dif_class(364)
