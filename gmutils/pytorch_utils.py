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

torch.set_printoptions(linewidth=260)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INF    = torch.Tensor([float("Inf")]).sum().to(DEVICE)
negINF = torch.Tensor([float("-Inf")]).sum().to(DEVICE)

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

    sys.stderr.write("\nINFO from file: %s"% file + " Line: %d"% line + "\n\tsize: %s"% str(T.size()) + "\n\ttype: %s\n"% str(type(T)))
    sys.stderr.write("\n\type: %s\n"% str(T.type()))
    try:
        sys.stderr.write("\tdtype: %s\n"% str(T.data.type()))
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
        
    T = T.to(DEVICE)
    V = torch.autograd.Variable(T, requires_grad=requires_grad)
    
    return V


def var_zeros(n):
    """
    Returns Variable ready for computational graph
    """
    T = torch.zeros(n)
    T = T.to(DEVICE)
    V = torch.autograd.Variable(T, requires_grad=False)
    
    return V
    

def var_ones(n):
    """
    Returns Variable ready for computational graph
    """
    T = torch.ones(n)
    T = T.to(DEVICE)
    V = torch.autograd.Variable(T, requires_grad=False)
    
    return V


def empty():
    """
    Return a sequence of precisely zero tensors
    """
    return var_zeros(0)


def learning_rate_by_epoch(epoch, lr):
    """
    To compute a new learning rate for each epoch (lower each time, of course)

    Parameters
    ----------
    epoch : int

    Returns
    -------
    float

    """
    return lr * (0.8 ** (epoch-1))


def hasnan(T):
    """
    Determine if a tensor has a NaN or Inf in it
    """
    s = T.data.sum()
    if s != s:
        return True
    if s == INF:
        return True
    if s == negINF:
        return True

    T = T.data.cpu()
    result = (T != T).numpy()
    if result.sum():
        err([s])
        return True
    return False


def squash(T):
    """
    Normalize length of vector to the range [0,1] without altering direction.
    """
    if not T.sum().gt(0):
        return T

    sq = T.pow(2)
    if not sq.sum().gt(0):
        return T

    sqnorm = sq.sum(-1, keepdim=True)
    if not sqnorm.sum().gt(0):  err(); exit()

    denom = 1 + sqnorm
    scale = sqnorm / denom
    unitvec = T / torch.sqrt(sqnorm)
    out = sqnorm * unitvec

    return out


################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})


################################################################################
################################################################################
