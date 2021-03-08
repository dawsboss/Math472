#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:40:57 2021

@author: grant
3.9
"""
import math
#import NewtonMethod.py as NM
#from NewtonMethod.py import Better_Error_Newton_Method as NM

def derivatice_approx( f, x, h ):
    return ( 8*f(x+h) - 8.0*f(x-h) - f( x + 2.0*h) + f(x - 2.0*h) ) / (12.0 * h)

def NM( f, x0, err, df=False, verbose = False):
    rtn = x0
    count = 0
    old = 0.0
    
    if(df):
        if verbose:
            print(f"while: {(abs(f(rtn))) + (abs(rtn-old))} < {err/5}")
        while((abs(f(rtn))) + (abs(rtn-old)) > err/5):
            count = count +1
            old = rtn
            rtn = rtn - (f(rtn) / df(rtn))
            if(verbose):
                print(f"count: {count} | rtn: {rtn}\n")
            
    else:
        df = derivatice_approx
        if verbose:
            print(f"while: {(abs(f(rtn))) + (abs(rtn-old))} < {err/5} => {(abs(f(rtn))) + (abs(rtn-old)) < err/5}")
        while((abs(f(rtn))) + (abs(rtn-old)) > err/5):
            count = count +1            
            old = rtn
            if df(f, rtn, .000000001) == 0:
                return ('Error, too flat', count)
            rtn = rtn - (f(rtn) / df(f, rtn, .000000001))
            if(verbose):
                print(f"count: {count} | rtn: {rtn}\n")
    
    return (rtn,count)


#5. For each function listed below, find an interval [a,b] such that g([a,b]) C [a,b].
#  Draw a graph of y = g(x) and y = x over this interval, and confirm that a fixed point
#  exists there. Estimate (by eye) the value of the fixed point, and use this as a starting
#  value for a fixed-point iteration. Does the iteration converge? Explain.


def fixed_point (x0, f, n, verbose=False):
    retval = x0    
    if verbose:
        print(f"Init retval : {retval}")
    for i in range(n):
        retval = f(retval)
        if verbose:
            print(f"Iteration #{i} has value: {retval}")
    return retval


#3.9.5 
A = lambda x: .5*(x + (2/x))
B = lambda x: x + math.pow(math.e, -x) - (1/4)
C = lambda x: math.cos(x)
D = lambda x: 1 + math.pow(math.e, -x)
E = lambda x: .5*(1 + x**2)

nmA = lambda x: x - (.5*(x + (2/x)))
nmB = lambda x: x - (x + math.pow(math.e, -x) - (1/4))
nmC = lambda x: x - (math.cos(x))
nmD = lambda x: x - (1 + math.pow(math.e, -x))
nmE = lambda x: x - (.5*(1 + x**2))

err = 10**-10
itter = 100

# Here we are going to compare Newtons method to fixed point 



print("A:")
x0 = 1.5
fpA = fixed_point(x0, A, itter, verbose = False)
print(f"Floating Point est of A: {fpA}")
fpNA = NM(nmA, x0, err, verbose=False)
print(f"Newton's Method estimation of A: {fpNA}")
  
print("B:")
x0 = 1.5
fpB = fixed_point(x0, B, itter, verbose = False)
print(f"Floating Point est of A: {fpB}")
fpNB = NM(nmB, x0, err, verbose=False)
print(f"Newton's Method estimation of A: {fpNB}")

print("C:")
x0 = .75
fpC = fixed_point(x0, C, itter, verbose = False)
print(f"Floating Point est of A: {fpC}")
fpNC = NM(nmC, x0, err, verbose=False)
print(f"Newton's Method estimation of A: {fpNC}")

print("D:")
x0 = 1.25
fpD = fixed_point(x0, D, itter, verbose = False)
print(f"Floating Point est of A: {fpD}")
fpND = NM(nmD, x0, err, verbose=False)
print(f"Newton's Method estimation of A: {fpND}")

print("E:")
x0 = .99
fpE = fixed_point(x0, E, 1000, verbose = False)
print(f"Floating Point est of A: {fpE}")
fpNE = NM(nmE, x0, err, verbose=False)
print(f"Newton's Method estimation of A: {fpNE}")