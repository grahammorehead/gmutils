""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gmutils.utils import err, argparser, isTrue, read_dir

torch.set_printoptions(linewidth=260)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INF    = torch.Tensor([float("Inf")]).sum().double().to(DEVICE)
negINF = torch.Tensor([float("-Inf")]).sum().double().to(DEVICE)

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
    sys.stderr.write("\tType: %s\n"% str(T.type()))
    try:
        sys.stderr.write("\tDType: %s\n"% str(T.data.type()))
    except:
        pass
    print()
    

def torchvar(X, ttype=torch.DoubleTensor, requires_grad=False):
    """
    Converts X into a PyTorch autograd Variable, ready to be part of a computational graph
    """
    if len(X) > 1:
        narr = np.array(X)
        
        T = torch.from_numpy(narr)
        if ttype == torch.DoubleTensor:
            T = T.double()
        elif ttype == torch.FloatTensor:
            T = T.float()
    else:
        T = ttype(X)
        
    T = T.to(DEVICE)
    V = torch.autograd.Variable(T, requires_grad=requires_grad)
    
    return V


def var_zeros(n, ttype=torch.DoubleTensor):
    """
    Returns Variable ready for computational graph
    """
    T = torch.zeros(n)
    if ttype == torch.DoubleTensor:
        T = T.double()
    elif ttype == torch.FloatTensor:
        T = T.float()
            
    T = T.to(DEVICE)
    V = torch.autograd.Variable(T, requires_grad=False)
    
    return V
    

def var_ones(n, ttype=torch.DoubleTensor):
    """
    Returns Variable ready for computational graph
    """
    T = torch.ones(n)
    if ttype == torch.DoubleTensor:
        T = T.double()
    elif ttype == torch.FloatTensor:
        T = T.float()
            
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
    return lr * (0.9 ** (epoch-1))


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


def model_files_by_loss(dirpath):
    """
    List all available model files by loss
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        if re.search('^L', f):
            lv = re.sub(r'^L', '', f)
            lv = re.sub(r'_E\d+$', '', lv)
            loss_val = float(lv)
            models[loss_val] = dirpath +'/'+ f

    return sorted(models.items(), key=lambda x: x[0])

    
def model_files_by_timestamp(dirpath):
    """
    List all available model files by timestamp, most recent first
    """
    models = {}
    model_files = read_dir(dirpath)
    for f in model_files:
        if re.search('^L', f):
            filepath = dirpath +'/'+ f
            ts = os.path.getmtime(filepath)
            models[ts] = filepath

    return sorted(models.items(), key=lambda x: x[0], reverse=True)  # "reverse", because we want the highest timestamps (most recent) first


def beta_choose(N):
    """
    Use an almost-Beta function to select an integer on [0, N]

    """
    x = int( np.random.beta(1, 128) * N+1 )
    if x > 0:
        x -= 1
    return x


def get_good_model(dirpath):
    """
    Randomly select and retrieve a good model (based on its loss) via a Beta distribution.  This will usually select the model correlated
    with the lowest loss values but not always.  May help to escape from local minima.  Although the currently selected loss function is
    convex for any given batch, since the graph is redrawn for each batch, I cannot yet confirm global convexity

    Returns
    -------
    dict { var_name : numpy array }
        Use these values to assign arrays to tensors

    """
    try:
        models = model_files_by_loss(dirpath)   # sorted from lowest loss to highest
        x = beta_choose(len(models))
        loss_val, filepath = models[x]
        model = json_load_gz(filepath)
        return model
    except:
        return None


def get_recent_model(dirpath):
    """
    Randomly select and retrieve a recent model via a Beta distribution.  This will usually select the most recent model but not always.
    May help to escape from local minima.  Although the currently selected loss function is convex for any given batch, since the graph
    is redrawn for each batch, I cannot yet confirm global convexity

    A separate process will attempt to winnow out the models with higher loss.

    Returns
    -------
    dict { var_name : numpy array }
        Use these values to assign arrays to tensors

    """
    try:
        models = model_files_by_timestamp(dirpath)   # sorted, most recent first
        x = beta_choose(len(models))
        loss_val, filepath = models[x]
        model = json_load_gz(filepath)
        return model
    except:
        return None

    
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})


################################################################################
################################################################################
