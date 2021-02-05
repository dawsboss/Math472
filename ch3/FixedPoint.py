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
            if df(f, rtn, .000001) == 0:
                return ('Error, too flat', count)
            rtn = rtn - (f(rtn) / df(f, rtn, .000000001))
            if(verbose):
                print(f"count: {count} | rtn: {rtn}\n")
    
    return (rtn,count)





















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

nmA = lambda x: .5*(x + (2/x))
nmB = lambda x: x + math.pow(math.e, -x) - (1/4)
nmC = lambda x: math.cos(x)
nmD = lambda x: 1 + math.pow(math.e, -x)
nmE = lambda x: .5*(1 + x**2)

err = 10**-10

# Here we are going to compare Newtons method to fixed point 

print("A:")
x0 = 1.5
fpA = fixed_point(x0, A, 100, verbose = False)
print(f"Floating Point est of A: {fpA}")
#fpNA = NM(nmA, 1.4, err, verbose=True)
#print(f"Newton's MEthod estimation of A: {fpNA}")
  
print("B:")
x0 = 1.5
fpB = fixed_point(x0, B, 100, verbose = False)
print(f"Floating Point est of A: {fpB}")
#fpNA = NM(nmA, 1.4, err, verbose=True)
#print(f"Newton's MEthod estimation of A: {fpNA}")

print("C:")
x0 = .75
fpC = fixed_point(x0, C, 100, verbose = False)
print(f"Floating Point est of A: {fpC}")
#fpNA = NM(nmA, 1.4, err, verbose=True)
#print(f"Newton's MEthod estimation of A: {fpNA}")

print("D:")
x0 = 1.25
fpD = fixed_point(x0, D, 100, verbose = False)
print(f"Floating Point est of A: {fpD}")
#fpNA = NM(nmA, 1.4, err, verbose=True)
#print(f"Newton's MEthod estimation of A: {fpNA}")

print("E:")
x0 = 1
fpE = fixed_point(x0, E, 100, verbose = False)
print(f"Floating Point est of A: {fpE}")
#fpNA = NM(nmA, 1.4, err, verbose=True)
#print(f"Newton's MEthod estimation of A: {fpNA}")