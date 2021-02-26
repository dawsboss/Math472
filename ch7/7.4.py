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


A = np.matrix([ [14, 14, -9, 3, -5],
                [14, 52, -15, 2, -32],
                [-9, -15, 36, -5, 16],
                [3, 2, -5, 47, 49],
                [-5, -32, 16, 49, 79]], dtype=float)
B = np.matrix([[-15],[-100],[106],[329],[463]], dtype=float)

(LUfact, idx) = ch7.LU(A)
x = ch7.LUsolver(LUfact, B, idx)
print(LUfact)
print(idx)
print(x)