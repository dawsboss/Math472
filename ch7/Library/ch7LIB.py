#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:24:10 2021

@author: grant
Library for chapter 7
"""
import numpy as np
import math

#Hn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Hn-style Matrix
def HnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            rtn[i,j] = 1.0/( i+j+1 )
    return rtn

#Kn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Kn-style Matrix
def KnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 2
            elif abs(i-j)==1:
                rtn[i,j]= -1
            else:
                rtn[i,j] = 0
    return rtn

#Tn Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy Tn-style Matrix
def TnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 4
            elif abs(i-j)==1:
                rtn[i,j] = 1
            else:
                rtn[i,j] = 0
    return rtn

#An Matrix creator function:
#   Parameters: 
#       n - size of square matrix
#   Outputs:
#       numpy An-style Matrix
def AnMatrix( n ):
    rtn = np.zeros((n,n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i==j:
                rtn[i,j] = 1
            elif i-j==1:
                rtn[i,j] = 4
            elif i-j==-1:
                rtn[i,j] = -4
            else:
                rtn[i,j] = 0
    return rtn


#7.2 - Alg 7.1 - for Ax=b for an A that is square 
#   Parameters:
#       A - numpy matrix
#       B - numpy matrix
#   Output:
def GaussianElim(A,B):
    n = A.shape[0]
    X = np.copy(B)
    for i in range(1, n):
        for j in range(i+1, n+1):
            m = A(j,i)/A(i,i)
            for k in range(i+1, n):
                A[j,k] = A[j,k] - (m*A[i,k])
            B[j] = B[j] - m*B[i]
    return B


#Used for testing the library code
if __name__ == '__main__':
    B = HnMatrix(5)
    print(B)
    print("I am stray lib code")