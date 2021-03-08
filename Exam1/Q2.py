#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:27:04 2021

@author: grant
"""

import math
import numpy as np
import matplotlib.pyplot as plt

def SnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(1,n+1):
        for j in range(1,n+1):
            if i==j:
                rtn[i-1,j-1] = 1.0/(i+j-1)
            elif abs(i-j)==1:
                rtn[i-1,j-1] = 1.0/(i+j-1)
    return rtn

def GausElimPP(A_,B_,verbose=False):
    A = np.copy(A_)
    b = np.copy(B_)
    n = A.shape[0]
    for i in range(n-1):
        am = abs(A[i,i])
        p = i
        for j in range(i+1, n):
            if abs(A[j,i]) > am:
                am = abs(A[j,i])
                p = j
        if p > i:
            for k in range(i, n):
                hold = A[i,k]
                A[i,k] = A[p,k]
                A[p,k] = hold
            hold = np.copy(b[i])
            b[i] = np.copy(b[p])
            b[p] = hold
        for j in range(i+1, n):
            m = A[j,i]/A[i,i]
            for k in range(i+1, n):
                A[j,k] = A[j,k] - m*A[i,k]
            b[j] = b[j] - m*b[i]
    return A, b
            
def GaussianPT2(A_,B_, verbose=False):
    A=np.copy(A_)
    B=np.copy(B_)
    n = A.shape[0]
    X = np.zeros((n,1))
    X[n-1] = B[n-1] / A[n-1,n-1]
    for i in range(n-2, -1, -1):
        sum = 0
        for j in range(i, n):
            sum = sum + A[i,j]*X[j]
        X[i] = (B[i]-sum)/A[i,i]
    return X


A = SnMatrix(10)
B = np.matrix([[1],[1],[1],[1],[1],[1],[1],[1],[1],[1]])

Ahat, Bhat = GausElimPP(A,B)
x = GaussianPT2(Ahat, Bhat)
print("x:\n",x)