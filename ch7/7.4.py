#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grant

7.4 presentation
All source code in ch7LIB 
"""
import numpy as np
import math
import ch7.Library.ch7LIB as ch7


A = np.matrix([ [4, 2, 0],
                [2, 3, 1],
                [0, 1, 5.0/2.0]], dtype=float)
B = np.matrix([[2],[5],[6]], dtype=float)

(LUfact, idx) = ch7.LU(A)
x = ch7.LUsolver(LUfact, B, idx)
print(LUfact)
print(idx)
print(x)