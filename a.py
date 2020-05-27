import numpy as np
import scipy.optimize as opt

def func(x, N):
    f_value = 0.
### START YOUR CODE HERE ###
    for i in range(1,N+1):
        f_value += x**i
    f_value -= N/2
#### END YOUR CODE HERE ####
    return float(f_value)

def func_prime(x, N):
    fp_value = 0.
### START YOUR CODE HERE ###
    for i in range(1,N+1):
        fp_value += i*x**(i-1)
#### END YOUR CODE HERE ####
    return float(fp_value)

def find_the_root(N):
    solution = 0.5
### START YOUR CODE HERE ###
    opt.newton(func, 0, func_prime, args=[N])
    solution1 = []
    solution1.append()
#### END YOUR CODE HERE ####
    return float(solution)

