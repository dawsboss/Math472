#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: grant
7.2
"""
import math
import numpy as np
import ch7.Library.ch7LIB as ch7

A = np.matrix([[ 14, 14, -9, 3, -5, -15],
               [ 14, 52, -15, 2, -32, -100],
               [ -9, -15, 36, -5, 16, 106],
               [ 3, 2, -5, 47, 49, 329],
               [ -5, -32, 16, 49, 79, 463]], dtype=float)

B = np.matrix([[-15], [-100], [106], [329], [463]], dtype=float)

print("7.2.1:")
print(f"A:\n {A}\n")
print(f"B:\n {B}\n")

#A,B = ch7.GaussianPT1(A,B)
A,B,X = ch7.GausElim(A,B)

print(f"A:\n {A}\n")
print(f"B:\n {B}\n")

#X = ch7.GaussianPT2(A,B)
print(X)

print('\n')

#7.2.8
A = np.matrix([ [14, 14, -9, 3, -5],
		[14, 52, -15, 2, -32],
		[-9, -15, 36, -5, 16],
		[3, 2, -5, 47, 49],
		[-5, -32, 16, 49, 79]], dtype=float)
B = np.matrix([[-15],[-100],[106],[329],[463]])

print("7.2.8:")
print(f"A:\n {A}\n")
print(f"B:\n {B}\n")

#A,B = ch7.GausElim(A,B)
A,B = ch7.GaussianPT1(A,B)

print(f"A:\n {A}\n")
print(f"B:\n {B}\n")

X = ch7.GaussianPT2(A,B)
print(X)
