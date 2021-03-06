#!/usr/bin/env python3
# -*- Coding: UTF-8 -*
"""
Count Class function.

Return a dictionary where key represents the class and value the total number
of object in two scenes.

Example
-------

>>> from numpy.random import randint
>>> from multiresolutionfit import countclass
>>> scene1 = randint(256, size=(50, 50))
>>> scene2 = randint(256, size=(50, 50))
>>> cnt1, cnt2 = countclass(scene1, scene2)

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
"""

from collections import OrderedDict
from numpy import unique


def countclass(scene1, scene2):
    """
    Count Class function.

    Return a dictionary where key represents the class and value the total number
    of object in two scenes.

    :parameter 2d_array scene1: numpy gray image array.
    :parameter 2d_array scene2: numpy gray image array.


    Example
    -------

    >>> from numpy.random import randint
    >>> from multiresolutionfit import countclass
    >>> scene1 = randint(256, size=(50, 50))
    >>> scene2 = randint(256, size=(50, 50))
    >>> cnt1, cnt2 = countclass(scene1, scene2)
    """
    unq1, cnts1 = unique(scene1, return_counts=True)
    unq2, cnts2 = unique(scene2, return_counts=True)

    cnt1 = OrderedDict(zip(unq1, cnts1))
    cnt2 = OrderedDict(zip(unq2, cnts2))

    return cnt1, cnt2
