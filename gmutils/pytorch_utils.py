""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gmutils.utils import err, argparser, isTrue

################################################################################
# FUNCTIONS

def print_info(T):
    """
    Returns some info about a tensor or variable
    """
    callerframerecord = inspect.stack()[1]    # 0 represents this line
                                              # 1 represents line at caller
    frame = callerframerecord[0]
    info  = inspect.getframeinfo(frame)
    file  = os.path.basename(info.filename)
    line  = info.lineno

    print("\nINFO from file: %s"% file, " Line: %d"% line, "\n\tsize: %s"% str(T.size()), "\n\ttype: %s"% str(type(T)))
    try:
        print("\tdtype:", T.data.type())
    except:
        pass
    print()
    

def torchvar(X, ttype=torch.FloatTensor, requires_grad=False):
    """
    Converts X into a PyTorch autograd Variable, ready to be part of a computational graph
    """
    if len(X) > 1:
        narr = np.array(X)
        
        T = torch.from_numpy(narr)
        if ttype == torch.FloatTensor:
            T = T.float()
    else:
        T = ttype(X)
        
    V = torch.autograd.Variable(T, requires_grad=requires_grad)
    
    return V


def var_zeros(n):
    """
    Returns Variable ready for computational graph
    """
    T = torch.zeros(n)
    V = torch.autograd.Variable(T, requires_grad=False)
    return V
    

def var_ones(n):
    """
    Returns Variable ready for computational graph
    """
    T = torch.ones(n)
    V = torch.autograd.Variable(T, requires_grad=False)
    return V


def empty():
    """
    Return a sequence of precisely zero tensors
    """
    return var_zeros(0)
    

################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})


################################################################################
################################################################################






                          
