#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:27:27 2021

@author: grant
"""

import numpy as np
import math
import ch7.Library.ch7LIB as ch7





K = np.matrix([[2,-1],[-1,2]],dtype=float)
def phi(x):#x must be a 2x1 numpy matrix
    a = (1/9) * math.pow(math.e, (-x[0,0]))
    b = (1/9) * math.pow(math.e, (-x[1,0]))
    return np.matrix([[a],[b]], dtype=float)
b = np.matrix([[-1],[1]],dtype=float)

print(b[0,0])

rtn = np.matrix([[1],[1]], dtype=float)
for _ in range(1,3):
    prev = rtn
    rtn = .5*(np.subtract(b, phi(np.add( np.subtract(rtn, np.matmul(K, rtn)) , 2*rtn)) ))
    print(rtn)
print(f"Done | rtn: \n{rtn}")

print("Part 2\n")
input()

Kinv = np.linalg.inv(K)

rtn2 = np.matrix([[1],[1]], dtype=float)
for _ in range(1,3):
    prev = rtn2
    rtn2 = np.matmul(Kinv, np.subtract(b, phi(rtn)))
print(f"Done | rtn2: \n{rtn2}")






"""
A = lambda x: x**2
B = lambda x,y: x**3 + 3*y 

# a.__code__.co_argcount
# a.__code__.co_varnames

f = [A, B]
print(f[1](1,2))


import sympy as sy
from sympy.utilities.lambdify import lambdify, implemented_function


#sy.init_printing()#This is to make a pretty, clean, and readable output

x = sy.symbols("x")#This is to say x is a sympy varriable 

f = x**2

print(f.subs(x,8))

dx = sy.Derivative(f)
dx = dx.doit()
print(dx)
print(dx.subs(x,8))

M = sy.Matrix([f, dx])
print(M)
print(M[0])
#print(M[0].sub(x,5))

w = [[f],[dx]]
print(w[0][0].subs(x,4))
print(len(w))

def ree (y):
    return y**5
    
F = implemented_function('F', ree)
#do stuff
lam_f = lambdify(x, F(x))
print(lam_f(4))

print("\n")

print(ree.__code__.co_argcount)
print(f.count(x))
print(f.n())
print(lam_f.__code__.co_argcount) # lam_f.__code__.co_argcount
#for i in len(tup):
#    print(f.args()[i])
#print(f.args())

print("\n")
print("testing")

#f = [A, B]
#c = ch7.KInv(f)
"""