#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:35:17 2021

@author: grant
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def fixed_point (x0, f, n, verbose=False):
    retval = x0    
    if verbose:
        print(f"Init retval : {retval}")
    for i in range(n):
        retval = f(retval)
        if verbose:
            print(f"Iteration #{i} has value: {retval}")
    return retval



F = lambda x: -3*x + 2*math.exp(-x**2)

alpha = fixed_point(.5, F, 100)
print(alpha)