#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 14:16:59 2021

@author: grant
"""
import numpy as np
import ch8.Library.ch8LIB as ch8

A = np.matrix([[1,2,3,4],
                [2,6,2,1],
                [0,0,0,0],
                [1,1,2,6]],dtype=float)
print(A)
print()
X,P = ch8.HessenbergItter(A, verbose=True)
print(X)
print(P)
print(np.matmul(P, np.matmul(X, P.T)))
"""
input("********* Next *********")

A2 = np.matrix([[6,1,1,1],
                [1,6,1,1],
                [1,1,6,1],
                [1,1,1,6]],dtype=float)
print("\n*** New Ex *** \n")
print(A2)
print()
(X2, Q2) = ch8.HessenbergItter(A2, verbose=True)
print(X2)
print()
#print(Q2)
print()

from scipy.linalg import hessenberg
print("Testing answers with spicy")
H, Q = hessenberg(A2, calc_q=True)
print(H)
print(Q)
print(Q2)

print(np.matmul(Q, np.matmul(H, Q.T)))
print(np.matmul(Q2, np.matmul(X2, Q2.T)))
"""