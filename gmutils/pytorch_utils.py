""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re
import random
import shutil
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

    try:
        size = str(T.size())
        typ  = str(type(T))
        Ttyp = str(T.type())
    except:
        size = "None"
        typ  = "None"
        Ttyp = "None"
    
    sys.stderr.write("\nINFO from file: %s"% file + " Line: %d"% line + "\n\tsize: %s"% size + "\n\ttype: %s\n"% typ)
    sys.stderr.write("\tType: %s\n"% Ttyp)
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
    return lr * (0.8 ** (epoch-1))


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
    return lt * (0.9 ** (epoch-1))


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
        lv = re.sub(r'^L', '', f)
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

    
def clear_chaff_by_loss(dirpath):
    """
    Get a list of all models sorted by loss.  Keep MAX, and delete the rest
    """
    verbose = False
    MAX = 100

    models = model_files_by_loss(dirpath)   # sorted, lowest loss first
    if len(models) > MAX:
        try:
            to_delete = models[MAX:]
            for model in to_delete:
                filepath, loss_val = model
                if verbose:
                    t = file_timestamp(filepath)
                    print("rm -fr (%s):"% t, filepath)
                shutil.rmtree(filepath)
        except:
            pass

    
def beta_choose(N):
    """
    Use an almost-Beta function to select an integer on [0, N]

    """
    x = int( np.random.beta(1, 8) * N+1 )
    if x > 0:
        x -= 1
    return x


def good_model_file_by_loss(dirpath):
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
        filepath, val = models[x]
        return filepath
    except:
        raise


def recent_model_file(dirpath):
    """
    Randomly select and retrieve a recent model via a Beta distribution.  This will usually select the most recent model but not always.
    May help to escape from local minima.  Although the currently selected loss function is convex for any given batch, since the graph
    is redrawn for each batch, I cannot yet confirm global convexity

    Returns
    -------
    str
    """
    try:
        models = model_files_by_timestamp(dirpath)   # sorted, most recent first
        x = beta_choose(len(models))
        val, filepath = models[x]
        return filepath
    except:
        raise


def most_recent_model_file(dirpath):
    """
    Select and retrieve most recent model file

    Returns
    -------
    str
    """
    try:
        models = model_files_by_timestamp(dirpath)   # sorted, most recent first
        val, filepath = models[0]
        return filepath
    except:
        raise


def study_imbalanced_classes(labels, _monitor):
    """
    Use for determining just how imbalanced the labels are.  This information will be used to make weights for the cross entropy loss
    """
    if _monitor.get('W') is None:
        W = 0.0
        _monitor['W seen']  = 1
    else:
        W = _monitor['W']
        _monitor['W seen'] += 1
        
    total    = float(torch.numel(labels))
    num_pos  = float(len(labels.data.nonzero()))
    W       += num_pos/total
    W_norm   = W / _monitor['W seen']
    
    _monitor['W'] = W
    sys.stderr.write("\tW: %0.4f\n"% W_norm)


def null_tensor(X):
    """
    If a tensor is None or 0-dim
    """
    if X is None:
        return True
    try:
        if X.size() == torch.Size([0]):
            return True
    except:
        return True

    return False

    
def tensor_cat(A, B, dim=0):
    """
    Intelligent concatentation
    """
    if null_tensor(A)  and  null_tensor(B):
        return None
    if null_tensor(A):
        return B
    if null_tensor(B):
        return A
    return torch.cat([A, B], dim)

            
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})


################################################################################
################################################################################
