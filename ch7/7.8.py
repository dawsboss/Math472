#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 20:27:27 2021

@author: grant
"""

import numpy as np
import math
import ch7.Library.ch7LIB as ch7


A = lambda x: x**2
B = lambda x,y: x**3 + 3*y


f = [A, B]
print(f[1](1,2))


import sympy as sy
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
print(M[0].sub(x,5))


