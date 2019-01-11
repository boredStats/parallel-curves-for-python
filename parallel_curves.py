# -*- coding: utf-8 -*-
"""
Transcribed parallel curves function

See links for more:
    https://www.mathworks.com/matlabcentral/fileexchange/27873-parallel-curves
    https://www.mathworks.com/matlabcentral/fileexchange/8782-vector-algebra-for-arrays-of-any-size-with-array-expansion-enabled
Created on Fri Jan 11 11:10:28 2019
"""

import numpy as np

def magn(a, dim):
    """
    Support function for parallel curves, copying docstring from matlab function
    
    MAGN   Magnitude (or Euclidian norm) of vectors.
    If A is a vector (e.g. a P?1, 1?P, or 1?1?P array):
    
    B = MAGN(A) is a scalar (1?1), equal to the magnitude of vector A.
    
    B = MAGN(A, DIM) is eqivalent to MAGN(A) if DIM is the non-singleton 
    dimension of A; it is equal to A if DIM is a singleton dimension.
    
    If A is an N-D array containing more than one vector:
    
    B = MAGN(A) is an N-D array containing the magnitudes of the vectors
    constructed along the first non-singleton dimension of A.
    
    B = MAGN(A, DIM) is an N-D array containing the magnitudes of the
    vectors constructed along the dimension DIM of A.
    """
    dim = dim-1
    b = np.sum(np.conj(a) * a, axis=dim)
    return np.sqrt(b)

def parallel_curves(x, y, d=1, make_plot=False, flag1=True):
    """
    Copying documentation from matlab function
    
    Description

    Calculates the inner and outer parallel curves to the given x, y
    coordinate pairs.  By default the inner parallel is toward the center 
    of curvature while the outer parallel is away from the center of 
    curvature.  Use  flag1=0 to make the parallels stay on opposite sides 
    of the curve.  Input the x and y coordinate pairs, distance between 
    the curve and the parallel, and whether to plot the curves.
    
    Program is currently limited to rectangular coordinates.
    Attempts to make sure the parellels are always inner or outer.
    The inner parallel is toward the center of curvature
    while the outer parallel is away from the center of curvature.
    If the radius of curvature become infinite adn the center of curvature 
    changes sides then the parallels will switch sides.  If the parallels 
    should stay on teh sae sides then set flag1=0 to keep the parallels
    on the sides.  
    
    Implements "axis equal" so that the curves appear with equal
    scaling.  If this is a problem, type "axis normal" and the scaling goes
    back to the default.  This will have to be done for every plot or feel
    free to modify the program.
    
    My notes:
        Distance calculations are done in unit space. Recommend converting
        inputs into unit distance if they aren't already.
        
        For verification purposes, the matlab code has been transcribed
        into python as faithfully as possible.  
    
    Parameters
    ----------
    x : numpy array
        vector of x-coordinates
    
    y : numpy array
        vector of y-coordinates
        x and y should describe a 2nd-order polynomial curve.
    
    Optional
    --------
    d : int
        distance between original curve and calculated curve
        Default = 1
    
    make_plot : bool
        should plots be drawn?
    
    flag1 : bool
        should parallel curves stay on opposite side of the curve?
    
    Returns
    -------
    x_inner, y_inner, x_outer, y_outer, R, unv, concavity, overlap
        Coordinates/parameters for the calculated parallel curves
    """
    
    if x.ndim != 1 or y.ndim != 1:
        raise Exception ("X and Y must be vectors")
    
    dx = np.gradient(x)
    dy = np.gradient(y)
    
    dx2 = np.gradient(dx)
    dy2 = np.gradient(dy)
    
    nv = np.ndarray(shape=[len(dy), 2])
    nv[:, 0] = dy
    nv[:, 1] = -dx
    
    unv = np.zeros(shape=nv.shape)
    norm_nv = magn(nv, 2)
    
    unv[:, 0] = nv[:, 0] / norm_nv
    unv[:, 1] = nv[:, 1] / norm_nv
    
    r0 = (dx**2 + dy**2) ** 1.5
    r1 = np.abs(dx*dy2 - dy*dx2)
    R = np.divide(r0, r1)
    
    overlap = R < d
    
    dy3 = np.zeros(shape=dy2.shape)
    dy3[dy2 > 0] = 1
    concavity = 2 * dy3 - 1 
    
    if flag1==True:
        x_inner = x-unv[:, 0] * concavity * d
        y_inner = y-unv[:, 1] * concavity * d
        
        x_outer = x+unv[:, 0] * concavity * d
        y_outer = y+unv[:, 1] * concavity * d
    else:        
        x_inner = x-unv[:, 0] * d
        y_inner = y-unv[:, 1] * d
        
        x_outer = x+unv[:, 0] * d
        y_outer = y+unv[:, 1] * d

    res = {'x_inner': x_inner,
           'y_inner': y_inner,
           'x_outer': x_outer,
           'y_outer': y_outer,
           'R': R,
           'unv': unv,
           'concavity': concavity,
           'overlap': overlap}
    return res

if __name__ == "__main__":
    x = np.arange(1, 101)
    y = x ** 2
    res = parallel_curves(x, y, d=1, flag1=False)    