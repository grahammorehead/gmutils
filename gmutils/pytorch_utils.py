""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re
import random
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from gmutils.utils import err, argparser, isTrue, read_dir

################################################################################
# DEFAULTS

torch.set_printoptions(linewidth=260)
INF     = torch.Tensor([float("Inf")]).sum().double()
negINF  = torch.Tensor([float("-Inf")]).sum().double()

if torch.cuda.is_available():
    # torch.cuda.manual_seed_all(12345)
    cuda    = torch.device('cuda')
    INF     = INF.cuda()
    negINF  = negINF.cuda()

################################################################################
# VARS to be used in this file

INF = negINF = None

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

    if torch.cuda.is_available():
        T = T.cuda()
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
            
    if torch.cuda.is_available():
        T = T.cuda()
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
            
    if torch.cuda.is_available():
        T = T.cuda()
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


def loss_threshold_by_epoch(epoch, lt):
    """
    To compute a new learning rate for each epoch (lower each time, of course)

    Parameters
    ----------
    epoch : int

    Returns
    -------
    float

    """
    return lt * (0.95 ** (epoch-1))


def hasnan(T):
    """
    Determine if a tensor has a NaN or Inf in it
    """
    s = T.data.sum()
    if s != s:
        return True
    if s == get_inf():
        return True
    if s == get_neginf():
        return True

    T = T.data.cpu()
    result = (T != T).numpy()
    if result.sum():
        err([s])
        return True
    return False


def squash_verbose(T):
    """
    Normalize length of vector to the range [0,1] without altering direction.
    """
    print('T:', T)
    if not T.sum().gt(0):
        return T

    sq = T.pow(2)
    if not sq.sum().gt(0):
        return T
    print('sq:', sq)
    
    sqsum = sq.sum(-1, keepdim=True)
    if not sqsum.sum().gt(0):  err(); exit()
    print('sqsum:', sqsum)

    denom = 1 + sqsum
    print('denom:', denom)
    scale = sqsum / denom
    print('scale:', scale)
    unitvec = T / torch.sqrt(sqsum)
    print('unitvec:', unitvec)
    out = scale * unitvec

    return out


def squash(T):
    """
    Normalize length of vector to the range [0,1] without altering direction.
    """
    if not T.sum().gt(0):
        return T

    sq = T.pow(2)
    if not sq.sum().gt(0):
        return T
    
    sqsum = sq.sum(-1, keepdim=True)
    if not sqsum.sum().gt(0):  err(); exit()

    denom = 1 + sqsum
    scale = sqsum / denom
    unitvec = T / torch.sqrt(sqsum)
    out = scale * unitvec

    return out


def model_files_by_loss(dirpath):
    """
    List all available model files by loss
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        lv = re.sub(r'^F', '', f)
        lv = re.sub(r'_E\d+_B\d+$', '', lv)
        loss_val = float(lv)
        modelpath = dirpath +'/'+ f
        models[modelpath] = loss_val

    return sorted(models.items(), key=lambda x: x[1])

    
def model_files_by_F1(dirpath):
    """
    List all available model files by loss
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        fv = re.sub(r'^F', '', f)
        fv = re.sub(r'_E\d+_B\d+$', '', fv)
        F_val = float(fv)
        modelpath = dirpath +'/'+ f
        models[modelpath] = F_val

    return sorted(models.items(), key=lambda x: x[1], reverse=True)

    
def best_model_file_by_loss(dirpath):
    """
    List all available model files by loss
    """
    sorted_models = model_files_by_loss(dirpath)
    lowest_loss   = sorted_models[0][1]
    best_models   = []
    for model in sorted_models:
        if model[1] == lowest_loss:
            best_models.append(model[0])
            
    return random.choice(best_models)

    
def best_model_file_by_F1(dirpath):
    """
    List all available model files by loss
    """
    sorted_models = model_files_by_F1(dirpath)   # Highest-first
    lowest_loss   = sorted_models[0][1]
    best_models   = []
    for model in sorted_models:
        if model[1] == lowest_loss:
            best_models.append(model[0])
            
    return random.choice(best_models)

    
def model_files_by_timestamp(dirpath):
    """
    List all available model files by timestamp, most recent first
    """
    models = {}
    model_files = read_dir(dirpath)
    for f in model_files:
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
        # models = model_files_by_loss(dirpath)   # sorted from lowest loss to highest
        models = model_files_by_F1(dirpath)     # sorted from highest to lowest
        x = beta_choose(len(models))
        filepath, val = models[x]
        return filepath
    except:
        raise


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
        val, filepath = models[x]
        return filepath
    except:
        raise


def study_imbalanced_classes(labels, _monitor):
    """
    Use for determining just how imbalanced the labels are.  This information will be used to make weights for the cross entropy loss
    """
    if _monitor.get('W') is None:
        W = var_zeros(2)
    else:
        W = _monitor['W']
    total    = torch.numel(labels)
    num_pos  = labels.sum().item()
    num_neg  = total - num_pos
    w        = torchvar([1/num_neg, 1/num_pos])
    W        = W + w
    W_norm   = W / _monitor['i']

    print("\n\tW_norm =", W_norm)
    print()
    
    _monitor['W'] = W

    return _monitor

            
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})


################################################################################
################################################################################
