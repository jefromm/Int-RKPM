import numpy as np

"""
Some kernel functions- take in a vector of distances, evaluate the function, first, and second derivatives
"""

def SPLIN3(z):
    case1 = np.less_equal(z, 0.5)
    w1 = 2/3 - 4*z**2 + 4*z**3
    dwdr1 = -8*z + 12*z**2
    ddwddr1 = -8 + 24*z
    case2 = np.logical_and(np.less_equal(z,1.0), np.greater(z,0.5))
    w2= 4/3 - 4*z + 4*z**2 - (4*z**3)/3
    dwdr2 = -4 + 8*z -4*z**2
    ddwddr2 = 8 - 8*z
    w = np.multiply(case1,w1) + np.multiply(case2,w2)
    dwdr = np.multiply(case1,dwdr1) + np.multiply(case2,dwdr2)
    ddwddr = np.multiply(case1,ddwddr1) + np.multiply(case2,ddwddr2)
    return w, dwdr, ddwddr