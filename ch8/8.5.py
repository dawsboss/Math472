#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 10:31:46 2021

@author: grant
"""
import numpy as np
import ch8.Library.ch8LIB as ch8
import ch7.Library.ch7LIB as ch7

p = np.matrix([1, 5, -2, 1])
C1 = ch8.get_companion(p, rotated=False)
print(C1)
print(ch8.get_eigvals(C1))
