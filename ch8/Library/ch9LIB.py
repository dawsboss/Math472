#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 13:11:34 2021

@author: grant
"""

import numpy as np
import math
import sympy as sy
from sympy.utilities.lambdify import lambdify, implemented_function
import numpy.linalg as la
import ch8.Library.ch8LIB as ch8
import ch7.Library.ch7LIB as ch7

def explicitDiffusion(Uo, a, L, h, dt, T):
    hx = int(round(L/h))
    x = np.linspace(0, L, hx+1)
    nt = int(round(T/float(dt)))
    t = np.linspace(0, nt*dt, nt+1)
    u = np.zeros(hx+1)
    u_n = np.zeros(hx+1)
    F = (a*dt)/h**2
    
    #Alg start
    for i in range(hx+1):
        u_n[i] = Uo(x[i])
    for i in range(nt):
        for j in range(1, hx):
            u[j] = u_n[j] + F*(u_n[j-1] - 2*u_n[i] + u_n[i+1])
        u[0] = 0
        u[hx] = 0
        u_n[:] = u
    return u_n, x, t

def implicitDiffusion(Uo, a, L, dt, h, T):
    Nt = int(round(T/float(dt)))
    t=np.linspace(0, Nt*dt, Nt+1)
    Nx = int(round(L/h))
    x=np.linspace(0, L, Nx+1)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    u = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)
    F = (a*dt)/h**2
    A = np.zeros(Nx+1, Nx+1)
    B = np.zeros(Nx+1)
    
    for i in range(1, Nx):
        A[i,i-1] = -F
        A[i,i+1] = -F
        A[i,i] = 1+2*F
    A[0,0] = A[Nx,Nx] = 1
    for i in range(Nx+1):
        u_n[i] = Uo(x[i])
    
    pass
