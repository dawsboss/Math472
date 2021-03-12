#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:41:24 2021

@author: grant
"""
import numpy as np
import math
import ch7.Library.ch7LIB as ch7

A = np.matrix( [ [4, 1, 0, 0],
                 [1, 5, 1, 0],
                 [0, 1, 6, 1],
                 [1, 0, 1, 4] ], dtype=np.float64)

b = np.transpose(np.matrix([1, 7, 16, 14], dtype=np.float64))

x0 = np.transpose(np.matrix([0, 0, 0, 0], dtype=np.float64))

n=10

print(ch7.JacobiIteration(A,b,x0,n))

print(ch7.GaussSeidelIteration(A,b,x0,n))

w=1.9
print(ch7.SORIteration(A,b,x0,n,w))
