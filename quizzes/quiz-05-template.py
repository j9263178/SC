import numpy as np
import scipy.optimize as opt
from scipy.integrate import solve_ivp

def solve_puckpos(track):
    ret = np.zeros(2)
    ### START YOUR CODE HERE ###
    

    #### END YOUR CODE HERE ####
    return ret

if __name__ == '__main__':
    data = np.load('puck_data.npy')
    
    idx = np.random.randint(50)
    ret = solve_puckpos(data[idx])
    
    print('Ending position is at',ret)