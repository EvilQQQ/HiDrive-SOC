
"""
This file aims at buidling Lookup Tables for 1-D search
and 2-D search.
"""
import numpy as np
from scipy.optimize import curve_fit

def TableLookup1D(X_RNG, Y_RNG, XVal):
    '''
    1-D lookup Table:
    Input Para: X_RNG is a sorted (values from low to high) list.
    Input Para: Y_RNG is a list corresponding to the X_RNG list.
    Input Para: XVal is an input value for searching.
    '''
    #find the corresponding Y-value in this 1-D lookuptable
    #X-value will be adjusted in the range of the X list.
    XVal = max(XVal, X_RNG[0])
    XVal = min(XVal, X_RNG[-1])
    XPos = idx_match(X_RNG, XVal)
    y0 = Y_RNG[XPos]
    if XPos == len(X_RNG)-1:
        wgt = 0.
        return y0
    else:
        y1 = Y_RNG[XPos + 1]
        wgt = (XVal - X_RNG[XPos]) / (X_RNG[XPos+1] - X_RNG[XPos])
        return y0 + wgt * (y1 - y0)


def TableLookup2D(R_RNG, C_RNG, T_RNG, INPUT_RVal, INPUT_CVal):
    '''
    2-D lookup Table:
    Input Para: R_RNG is a sorted (values from low to high) list as rows in the 2-D Table.
    Input Para: C_RNG is a sorted (values from low to high) list as columns in the 2-D Table.
    Input Para: T_RNG is an 2-D array corresponding to the R_RNG list and the C_RNG list.
    Input Para: INPUT_RVal is an input value from the R_RNG list.
    Input Para: INPUT_CVal is an input value from the C_RNG list.
    '''
    r_idx = idx_match(R_RNG, INPUT_RVal)
    c_idx = idx_match(C_RNG, INPUT_CVal)
    max_row = len(R_RNG)-1
    max_col = len(C_RNG)-1

    if INPUT_CVal > C_RNG[-1] or INPUT_CVal < C_RNG[0]:
        # exponential curve to fit.
        pars, cov = curve_fit(f=func, xdata=C_RNG, ydata=T_RNG[r_idx,:])
        t_val0 = func(INPUT_CVal, pars[0], pars[1])

        if r_idx == max_row:
            t_val = t_val0
        else:
            pars, cov = curve_fit(f=func, xdata=C_RNG, ydata=T_RNG[r_idx+1,:])
            t_val1 = func(INPUT_CVal, pars[0], pars[1])
        
            r_wgt = (INPUT_RVal - R_RNG[r_idx]) / (R_RNG[r_idx+1] - R_RNG[r_idx])
            t_val = t_val0 + r_wgt * (t_val1 - t_val0)
    else:
        a = T_RNG[r_idx, c_idx]
        if r_idx == max_row and c_idx == max_col:
            t_val = a
        elif r_idx == max_row:
            c_wgt = (INPUT_CVal - C_RNG[c_idx]) / (C_RNG[c_idx+1] - C_RNG[c_idx])
            r_val0 = a
            r_val1 = T_RNG[r_idx, c_idx+1]
            t_val = r_val0 + c_wgt * (r_val1 - r_val0)
        elif c_idx == max_col:
            r_wgt = (INPUT_RVal - R_RNG[r_idx]) / (R_RNG[r_idx+1] - R_RNG[r_idx])
            r_val0 = a
            r_val1 = T_RNG[r_idx+1, c_idx]
            t_val = r_val0 + r_wgt * (r_val1 - r_val0)
        else:
            b = T_RNG[r_idx+1, c_idx]
            c = T_RNG[r_idx, c_idx+1]
            d = T_RNG[r_idx+1, c_idx+1]
    
            r_wgt = (INPUT_RVal - R_RNG[r_idx]) / (R_RNG[r_idx+1] - R_RNG[r_idx])
            c_wgt = (INPUT_CVal - C_RNG[c_idx]) / (C_RNG[c_idx+1] - C_RNG[c_idx])

            r_val0 = a + r_wgt * (b - a)
            r_val1 = c + r_wgt * (d - c)
            t_val = r_val0 + c_wgt * (r_val1 - r_val0)
    return t_val


def idx_match(array, value):
    '''
    This function is to find the highest number less than target value in a list.
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if array[idx]>value and idx>0:
        idx = idx - 1
    return idx

def func(x, a, b):
    return a * (b**x)

