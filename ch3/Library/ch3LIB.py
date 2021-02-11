#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 10:06:25 2021

@author: grant
Library for chapter 3
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def derivatice_approx( f, x, h ):
    return ( 8*f(x+h) - 8.0*f(x-h) - f( x + 2.0*h) + f(x - 2.0*h) ) / (12.0 * h)
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



#3.9
def fixed_point (x0, f, n, verbose=False):
    retval = x0
    
    if verbose:
        print(f"Init retval : {retval}")
    for i in range(n):
        retval = f(retval)
        if verbose:
            print(f"Iteration #{i} has value: {retval}")
    return retval