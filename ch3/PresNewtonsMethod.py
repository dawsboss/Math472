#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grant Dawson 
Math 472
3.2.2 / 3.2.3

2. Write down the iteration for Newton's method as applied to
 the function f(x) = x 3 — 2.
 Simplify the computation as much as possible. 
 What has been accomplished if we find the root of this function?

3. Generalize the preceeding two exercises by writing down the
 iteration for Newton's method as applied to f(x) — x n — a
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# This approximates the derivative of the given lambda functions 
def derivatice_approx( f, x, h ):
    return ( 8*f(x+h) - 8.0*f(x-h) - f( x + 2.0*h) + f(x - 2.0*h) ) / (12.0 * h)

# THis version of Newton's Method is for an input of an error to itterate to 
def Better_Error_Newton_Method( f, x0, err, df=False, verbose = True):
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
                print(f"count: {count} | n: {n}\n rtn: {rtn}\n")
            
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
                print(f"count: {count} | n: {n}\n rtn: {rtn}\n")
    
    return (rtn,count)

# This verssion for Newton's Method is for an inut of number of itteration to run
def Newton_Method( f, x0, n, df=False, verbose = False):
    rtn = x0
    if(df):
        for i in range(0,n+1):
            rtn = rtn - (f(rtn) / df(rtn))
    else:
        df = derivatice_approx
        for i in range(0,n+1):
            rtn = rtn - (f(rtn) / df(f, rtn, .000001))
    
    
    return rtn


# This is to graph Newton's Method
def Grapher_Newton_Method( f, x0, n, df=False, verbose = False):
    rtn = []
    rtn.append(x0)
    if(verbose):
        print(f"x0: {x0} | n: {n} | rtn: {rtn}\n")
    if(df):
        for i in range(0,n+1):
            rtn.append( rtn[i] - (f(rtn[i]) / df(rtn[i])) )
            if(verbose):
                print(f"i: {i} | n: {n}\n rtn: {rtn}\n")
    else:
        df = derivatice_approx
        for i in range(0,n+1):
            rtn.append( rtn[i] - (f(rtn[i]) / df(f, rtn[i], .000001)) )
            if(verbose):
                print(f"i: {i} | n: {n}\n rtn: {rtn}\n")  
    
    return rtn







FN = [
      lambda x: x**3 - 2,
      lambda x: x**2 - 2
     ]

xNot = [
        .000001,
        .001,
        1,
        1.5,
        2
        
       ]

names= [
        'x^3 - 2',
        'x^2 - 2'
       ]

err = 10**-6
n=0 # This will be used to store the number 
#       of itteration that was taken to find the root


for j in range(0,2):
    for i in xNot:
        name = names[j]
        func = FN[j]
        x0 = i
        result,n = Better_Error_Newton_Method(func, x0, err, verbose=False);
        print(f"f(x) = {name} | n: {n} | X0: {x0} |\
         Root: x = {result}");
    print()


         
input('Print part 2: ')

FN = [
      lambda x: x**3 - 1,
      lambda x: x**3 - 2,
      lambda x: x**3 - 3,
      lambda x: x**3 - 4,
      lambda x: x**3 - 5,
      lambda x: x**3 - 6,
      lambda x: x**3 - 7,
      lambda x: x**3 - 8,
      lambda x: x**3 - 9,
      lambda x: x**3 - 10,
     ]
names = [
        'x^3 - 1',
        'x^3 - 2',
        'x^3 - 3',
        'x^3 - 4',
        'x^3 - 5',
        'x^3 - 6',
        'x^3 - 7',
        'x^3 - 8',
        'x^3 - 9',
        'x^3 - 10',
        ]

for j in range(0,len(FN)):
    for i in xNot:
        name = names[j]
        func = FN[j]
        x0 = i
        result,n = Better_Error_Newton_Method(func, x0, err, verbose=False);
        print(f"f(x) = {name} | n: {n} | X0: {x0} |\
         Root: x = {result}");
    print()

input('Print part 3: ')



FN = [
      lambda x: x**3 - 2,
      lambda x: x**4 - 2,
      lambda x: x**5 - 2,
      lambda x: x**6 - 2,
      lambda x: x**7 - 2,
      lambda x: x**8 - 2,
      lambda x: x**9 - 2,
      lambda x: x**10 - 2,
      lambda x: x**11 - 2,
      lambda x: x**12 - 2,
     ]
names = [
        'x^3 - 2',
        'x^4 - 2',
        'x^5 - 2',
        'x^6 - 2',
        'x^7 - 2',
        'x^8 - 2',
        'x^9 - 2',
        'x^10 - 2',
        'x^11 - 2',
        'x^12 - 2',
        ]

for j in range(0,len(FN)):
    for i in xNot:
        name = names[j]
        func = FN[j]
        x0 = i
        result,n = Better_Error_Newton_Method(func, x0, err, verbose=False);
        print(f"f(x) = {name} | n: {n} | X0: {x0} |\
         Root: x = {result}");
    print()




