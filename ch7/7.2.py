#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:14:26 2021

@author: grant
7.2
"""
import math
import numpy as np
import Library.ch7LIB as ch7

A = np.matrix([[ 14, 14, -9, 3, -5, -15],
               [ 14, 52, -15, 2, -32, -100],
               [ -9, -15, 36, -5, 16, 106],
               [ 3, 2, -5, 47, 49, 329],
               [ -5, -32, 16, 49, 79, 463]], dtype=float)

B = np.matrix([[-15], [-100], [106], [329], [463]], dtype=float)

"""
A = np.matrix([ [2.0, -5.0],
		[-2.0, 4.0]])

B = np.matrix([[7.0], [-6.0]])
"""
"""
A = np.matrix([ [-4, 7, -2],
		[1, -2, 1],
		[2, -3, 1]], dtype=float)

B = np.matrix([[2],[3],[-1]], dtype=float)
"""
#A,B = ch7.GaussianPT1(A,B)

A,B = ch7.GausElim(A,B)

print()
print(A)
print()
print(B)
print()
print("TEST")
X = ch7.GaussianPT2(A,B)


print(X)
 
#B = ch7.HnMatrix(8)
#print(B)
