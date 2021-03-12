#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:41:24 2021

@author: grant
"""
import numpy as np
import math
import ch7.Library.ch7LIB as ch7

# A = np.matrix( [ [4, 1, 0, 0],
#                  [1, 5, 1, 0],
#                  [0, 1, 6, 1],
#                  [1, 0, 1, 4] ], dtype=np.float64)

# b = np.transpose(np.matrix([1, 7, 16, 14], dtype=np.float64))

# x0 = np.transpose(np.matrix([0, 0, 0, 0], dtype=np.float64))

# n=10

# print(ch7.JacobiIteration(A,b,x0,n))

# print(ch7.GaussSeidelIteration(A,b,x0,n))

# w=1.9
# print(ch7.SORIteration(A,b,x0,n,w))

print("\n7.7.7 goofing\n")

D = ch7.TnNegMatrix(4)
I = np.eye(4,dtype=np.float64)
Z = np.zeros(shape=(4,4))

A = np.bmat( [ [ D.copy(), I.copy(), Z.copy(), Z.copy() ],
                 [ I.copy(), D.copy(), I.copy(), Z.copy() ],
                 [ Z.copy(), I.copy(), D.copy(), I.copy() ],
                 [ Z.copy(), Z.copy(), I.copy(), D.copy() ] ] )
b = np.transpose(np.matrix([5, 11, 18, 21, 29, 40, 48, 48, 57, 72, 80, 76, 69, 87, 94, 85], dtype=np.float64))
errtol = 10e-10
x0 = np.zeros(shape=(16,1))

(x, count) = ch7.JacobiIterationErr(A,b,x0,errtol)
print(f"Jacobi Iteration converged in {count} moves\n") 

(x, count) = ch7.GaussSeidelIterationErr(A,b,x0,errtol)
print(f"Gauss converged in {count} moves\n") 

W = [.25, .5, .75, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
for w in W:
    (x, count) = ch7.SORIterationErr(A,b,x0,errtol,w)
    print(f"SOR converged in {count} moves | w:{w}") 
    
    
    
    
    
    
print("\n7.7.6\n")  
    
A = np.matrix( [ [ 4, -1, 0, 0 ],
                 [ -1, 4, -1, 0 ],
                 [ 0, -1, 4, -1 ],
                 [ -1, 0, -1, 4 ] ], dtype=np.float64 )
b = np.transpose(np.matrix([-1, 2, 4, 10],dtype=np.float64))
x0 = np.zeros(shape=(4,1))  
  
for w in W:
    (x, count) = ch7.SORIterationErr(A,b,x0,errtol,w)
    print(f"Jacobi Iteration converged in {count} moves | w:{w}") 