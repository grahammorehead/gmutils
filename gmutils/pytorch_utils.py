""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re, gc
import subprocess
import random
import shutil
import inspect
import math
import numpy as np
from gmutils.utils import err, argparser, isTrue, read_dir
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.modules.loss as Loss
    from torch.optim.optimizer import Optimizer

    ################################################################################
    # DEFAULTS

    torch.set_printoptions(linewidth=260)
    INF = torch.Tensor([float("Inf")]).sum().double()
    negINF = torch.Tensor([float("-Inf")]).sum().double()
    TORCH_DOUBLE = torch.DoubleTensor
    TORCH_LOSS = Loss._Loss
    L1_LOSS  = nn.L1Loss(reduction='sum')
    TORCH_ONE = TORCH_DOUBLE([1.])
    NEG_ONE   = TORCH_DOUBLE([-1.0])
    TORCH_TWO = TORCH_DOUBLE([2.])
    TORCH_E   = TORCH_DOUBLE([math.exp(1.)])
    TORCH_DILATION = TORCH_DOUBLE([517.])
    LEAKY_RELU = torch.nn.LeakyReLU(negative_slope=0.001, inplace=False)
    
    if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(12345)
        cuda    = torch.device('cuda')
        L1_LOSS = L1_LOSS.cuda()
        INF     = INF.cuda()
        negINF  = negINF.cuda()
        TORCH_ONE = TORCH_ONE.cuda()
        NEG_ONE   = NEG_ONE.cuda()
        TORCH_TWO = TORCH_TWO.cuda()
        TORCH_E = TORCH_E.cuda()
        TORCH_DILATION = TORCH_DILATION.cuda()
        LEAKY_RELU = LEAKY_RELU.cuda()
        
except Exception as e:
    TORCH_DOUBLE = None
    TORCH_LOSS = object
    raise
    err([], {'exception':e, 'level':0})

    
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
    grad_fn = T.grad_fn
    requires_grad = T.requires_grad
    
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
    sys.stderr.write("\tGrad Fn: %s\n"% grad_fn)
    sys.stderr.write("\tRequires grad: %s\n"% requires_grad)
    print()
    

def torchtensor(X, ttype=TORCH_DOUBLE, requires_grad=False):
    """
    Converts X into a PyTorch Tensor

    Parameters
    ----------
    X : in, float, or torch.Tensor

    """
    if isinstance(X, torch.Tensor):
        T = X
        if ttype == torch.DoubleTensor:    # float 64
            T = T.double()
        elif ttype == torch.FloatTensor:   # float 32
            T = T.float()
        elif ttype == torch.HalfTensor:    # float 16
            T = T.half()
        elif ttype == torch.ByteTensor:    # uint 8
            T = T.byte()
        elif ttype == torch.CharTensor:    # int 8
            T = T.char()
        elif ttype == torch.ShortTensor:   # int 16
            T = T.short()
        elif ttype == torch.IntTensor:     # int 32
            T = T.int()
        elif ttype == torch.LongTensor:    # int 64
            T = T.long()
            
    else:
        if isinstance(X, int)  or  isinstance(X, float):
            X = [X]
        if not isinstance(X, list):
            err([X])
            print("Problem making tensor.")
            print(ZZZ)
        T = ttype(X)

    T.requires_grad = requires_grad

    return T

    
def torchvar(X, ttype=TORCH_DOUBLE, requires_grad=False):
    """
    Converts X into a PyTorch tensor, ready to be part of a computational graph

    NOTE:  This code reflects a change in Pytorch 0.4.0, Variable and Tensor have merged

    Parameters
    ----------
    X : in, float, or torch.Tensor

    """
    T = torchtensor(X, ttype=ttype, requires_grad=requires_grad)   # First convert X to a tensor
    if torch.cuda.is_available():
        T = T.cuda()

    # Next line commented out for PyTorch 0.4.0  
    # V = torch.autograd.Variable(T, requires_grad=requires_grad)
    
    return T


def torchvar_list(X, ttype=TORCH_DOUBLE, requires_grad=False):
    """
    Converts a list X of Python variables into PyTorch tensors, ready to be part of a computational graph

    NOTE:  This code reflects a change in Pytorch 0.4.0, Variable and Tensor have merged
    """
    tensors = []
    for x in X:
        t = torchtensor(x, ttype=ttype, requires_grad=requires_grad)   # First convert X to a tensor
        tensors.append(t)
    T = torch.stack(tensors)
    if torch.cuda.is_available():
        T = T.cuda()
    
    return T


def var_zeros(n, ttype=TORCH_DOUBLE, requires_grad=False):
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
        
    T.requires_grad = requires_grad
    # V = torch.autograd.Variable(T, requires_grad=False)
    
    return T
    

def var_ones(n, ttype=TORCH_DOUBLE, requires_grad=False):
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
        
    T.requires_grad = requires_grad
    # V = torch.autograd.Variable(T, requires_grad=False)
    
    return T


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


def squash_old(T):
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


def squash(T, dim):
	"""
    This is Eq.1 from the CapsNet paper
    """
	mag_sq  = torch.sum(T**2, dim=dim, keepdim=True)
	mag     = torch.sqrt(mag_sq)
	T       = (mag_sq / (1.0 + mag_sq)) * (T / mag)
    
	return T


def loss_from_filename(filename):
    """
    Loss value gleaned from dirpath or filename
    """
    loss_val = None
    filename = re.sub(r'^.*(L\d)', r'\1', filename)
    if re.search(r'^L', filename):
        lv = re.sub(r'^L', '', filename)
        lv = re.sub(r'_E\d+_B\d+$', '', lv)
        try:
            loss_val = float(lv)
        except:
            err([lv])
            raise
        
    return loss_val

                
def model_files_by_loss(dirpath):
    """
    List all available model files by loss
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        if re.search(r'^L', f):
            lv = re.sub(r'^L', '', f)
            lv = re.sub(r'_E\d+_B\d+$', '', lv)
            try:
                loss_val = float(lv)
                modelpath = dirpath +'/'+ f
                models[modelpath] = loss_val
            except:
                pass

    return sorted(models.items(), key=lambda x: x[1])

    
def model_files_by_pcc(dirpath):
    """
    List all available model files by PCC
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        if re.search(r'^PCC', f):
            lv = re.sub(r'^PCC', '', f)
            lv = re.sub(r'_E\d+_B\d+$', '', lv)
            try:
                loss_val = float(lv)
                modelpath = dirpath +'/'+ f
                models[modelpath] = loss_val
            except:
                pass

    return sorted(models.items(), key=lambda x: x[1], reverse=True)

    
def model_files_by_F1(dirpath):
    """
    List all available model files by loss
    """
    models = {}
    model_dirs = read_dir(dirpath)
    for f in model_dirs:
        if re.search(r'^F', f):
            fv = re.sub(r'^F', '', f)
            fv = re.sub(r'_E\d+_B\d+$', '', fv)
            try:
                F_val = float(fv)
                modelpath = dirpath +'/'+ f
                models[modelpath] = F_val
            except:
                pass

    return sorted(models.items(), key=lambda x: x[1], reverse=True)

    
def best_model_file_by_loss(dirpath):
    """
    Return model file with lowest loss
    """
    sorted_models = model_files_by_loss(dirpath)
    best = sorted_models[0][0]
    return best

    
def best_model_file_by_pcc(dirpath):
    """
    Return model file with highest PCC
    """
    sorted_models = model_files_by_pcc(dirpath)
    best = sorted_models[0][0]
    return best

    
def best_model_file_by_F1(dirpath):
    """
    Return model file with highest F1
    """
    sorted_models = model_files_by_F1(dirpath)   # Highest-first
    best = sorted_models[0][0]
    return best

    
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

    
def clear_chaff_by_pcc(dirpath, MAX=100):
    """
    Get a list of all models sorted by loss.  Keep MAX, and delete the rest
    """
    verbose = False

    models = model_files_by_pcc(dirpath)   # sorted, highest PCC first
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

    
def clear_chaff_by_loss(dirpath, MAX=100, verbose=False):
    """
    Get a list of all models sorted by loss.  Keep MAX, and delete the rest
    """
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


def good_model_file_by_pcc(dirpath):
    """
    Randomly select and retrieve a good model (based on its PCC) via a Beta distribution.  This will usually select the model correlated
    with the lowest loss values but not always.  May help to escape from local minima.  Although the currently selected loss function is
    convex for any given batch, since the graph is redrawn for each batch, I cannot yet confirm global convexity

    Returns
    -------
    dict { var_name : numpy array }
        Use these values to assign arrays to tensors

    """
    try:
        models = model_files_by_pcc(dirpath)   # sorted from highest to lowest
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


def tensor_sum_old(tensors):
    """
    Element-wise sum a set of tensors all having same dimensionality
    """
    output = tensors[0]
    if len(tensors) > 1:
        for tensor in tensors[1:]:
            output = output + tensor
    return output
    
    
def tensor_sum(tensors):
    """
    Element-wise sum a set of tensors all having same dimensionality

    Parameters
    ----------
    tensors : python list of PyTorch tensors
    """
    tensors = torch.stack(tensors)
    T   = torch.sum(tensors, 0)
    return T
    
    
def tensor_sum_of_vectors(vectors):
    """
    Element-wise sum a set of tensors all having same dimensionality

    Parameters
    ----------
    tensors : python list of PyTorch tensors
    """
    tensors = torchvar_list(vectors)   # A stacked array of tensors with the same dimensionality
    T       = torch.sum(tensors, 0)
    return T
    
    
def L1_norm_sum_of_vectors(vectors):
    """
    L1-Normalized Element-wise sum a set of vectors all having same dimensionality

    Parameters
    ----------
    tensors : python list of python vectors (which are also lists)
    """
    tensors = torchvar_list(vectors)   # A stacked array of tensors with the same dimensionality
    T       = torch.sum(tensors, 0)          # A single tensor with the original dimension
    T       = F.normalize(T.unsqueeze(0), 1).squeeze(0)
    return T
    
    
def L2_norm_sum_of_vectors(vectors):
    """
    L2-Normalized Element-wise sum a set of vectors all having same dimensionality

    Parameters
    ----------
    tensors : python list of python vectors (which are also lists)
    """
    tensors = torchvar_list(vectors)   # A stacked array of tensors with the same dimensionality
    T       = torch.sum(tensors, 0)          # A single tensor with the original dimension
    T       = F.normalize(T.unsqueeze(0), 2).squeeze(0)
    return T
    
    
def L1_norm_sum(tensors):
    """
    L1-Normalized Element-wise sum a set of tensors all having same dimensionality

    Parameters
    ----------
    tensors : python list of PyTorch tensors
    """
    tensors = torch.stack(tensors)
    T       = torch.sum(tensors, 0)
    T       = F.normalize(T.unsqueeze(0), 1).squeeze(0)
    return T
    
    
def L2_norm_sum(tensors):
    """
    L2-Normalized Element-wise sum a set of tensors all having same dimensionality

    Parameters
    ----------
    tensors : python list of PyTorch tensors
    """
    tensors = torch.stack(tensors)
    T       = torch.sum(tensors, 0)
    T       = F.normalize(T.unsqueeze(0), 2).squeeze(0)
    return T
    
    
def L1_norm(T):
    """
    L1-Normalize a tensor
    """
    T       = F.normalize(T.unsqueeze(0), 1).squeeze(0)
    return T
    
    
def L2_norm(T):
    """
    L2-Normalize a tensor
    """
    T       = F.normalize(T.unsqueeze(0), 2).squeeze(0)
    return T
    
    
def dilate_positive_error(preds, labels):
    """
    Increase the cost between preds and labels in the cases where the label is a positive example

    This funciton acts like a non-learning layer

    Parameters
    ----------
    preds, labels : tensors of equal size (probably 1-dimensional)

    Returns
    -------
    preds, labels : same tensors but modified
    """
    N = labels.numel()
    att = N * labels
    labels = att + labels
    preds = att * preds + preds
    
    return preds, labels


def dilate_error(preds, labels):
    """
    Increase the cost between preds and labels via a smooth funciton, exentuating error

    This funciton acts like a non-learning layer

    Parameters
    ----------
    preds, labels : tensors of equal size (probably 1-dimensional)

    Returns
    -------
    preds, labels : same tensors but modified
    """
    N = labels.numel()
    att = N * labels
    labels = att + labels
    preds = att * preds + preds
    
    return preds, labels


def pearson_coeff(X, Y):
    """
    The Pearson correlation coefficient is a measure of the linear correlation between two variables.

    PCC = cov(X,Y)/(stdev_X * stdev_Y)
    """
    vx   = X - torch.mean(X)
    vy   = Y - torch.mean(Y)
    pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    
    return pcc


def pearson_loss(X, Y):
    """
    Loss function based on the Pearson Correlation Coefficient
    """
    return 1.0 - pearson_coeff(X, Y)


def get_GPU_memory():
    """
    Return a list of how much memory is currently used by each GPU
    """
    try:
        dev = int(os.environ['CUDA_VISIBLE_DEVICES'])
    except:
        dev = 0
    # mem = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']).strip().split(b'\n')
    # mem = list(map(int, mem))
    # return mem[dev]
    mem = torch.cuda.memory_allocated(device=dev)
    return mem


def print_GPU_memstats(mem_max=None):
    """
    If mem_max is None, assume Tesla V100 16GB
    """
    if mem_max is None:
        mem_max = 17179869184   # in bytes
    try:
        dev = int(os.environ['CUDA_VISIBLE_DEVICES'])
    except:
        dev = 0
    mem    = torch.cuda.memory_allocated(device=dev)
    sys.stderr.write("\t< usage: %0.3f >     (%d out of %d)\n"% (mem/mem_max, mem, mem_max))
    

def generate_powerlaw_distro(w):
    """
    For some width 'w', generate a set of points following the desired kind of power-law distribution

    Using:  y = 1/x
    """
    x         =  2  # the starting point
    end       = 10
    interval  = (end - x) / w
    output    = []

    for i in range(w):
        output.append(1/x)
        x += interval

    if len(output) > 256:
        err()
        exit()
    output = torchvar(output)

    return output


def count_tensors(gpu_only=True):
    """
    Deletes the Tensors being tracked by the garbage collector.
    """
    total_size = 0
    for obj in gc.get_objects():
        try:
            dirs = dir(obj)
            if len(dirs) == 0:
                continue
            if dir(obj)[0] in ['LoadLibrary', '_LazyLoader__load', '_LazyCorpusLoader__args']:
                continue
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    total_size += obj.data.numel()
        except Exception as e:
            pass        
    # sys.stderr.write("GC tracking: %d\n"% total_size)


def collect_garbage():
    # sys.stderr.write("Collecting garbage ...\n")
    gc.collect()
    torch.cuda.empty_cache()
    # sys.stderr.write("Should be empty now.\n")
    count_tensors()
    # print_GPU_memstats()


def get_binary_losses(preds, labels, verbose=False):
    """
    For some set of preds and labels, assumed to be binary, get the overall L1 loss for each set of labels
    """
    # Some tensors to use
    zeros    = torch.zeros_like(labels)
    ones     = torch.ones_like(labels)
    mask     = binarize(labels).detach()
    antimask = binarize(labels, options={'reverse':True}).detach()

    preds_zero  = antimask * preds
    preds_one   = mask * preds
    labels_zero = antimask * labels
    labels_one  = mask * labels

    if verbose:
        print("PREDS:    ", preds.cpu().data.numpy().tolist()[100:200])
        print("PREDS 0:  ", preds_zero.cpu().data.numpy().tolist()[100:200])
        print("PREDS 1:  ", preds_one.cpu().data.numpy().tolist()[100:200])
        print("LABELS:   ", labels.cpu().data.numpy().tolist()[100:200])
        print("LABELS 0: ", labels_zero.cpu().data.numpy().tolist()[100:200])
        print("LABELS 1: ", labels_one.cpu().data.numpy().tolist()[100:200])
        
    zeroloss   = L1_LOSS(preds_zero, labels_zero)
    oneloss    = L1_LOSS(preds_one, labels_one)
    
    if verbose:
        err([zeroloss, oneloss])
    
    return zeroloss, oneloss
    
    
def get_binary_dilation(preds, labels):
    """
    For some set of preds and labels, assumed to be binary, get the overall L1 loss for each set of labels

    Returns
    -------
    ratio 'loss @ label=0' to 'loss @ label=1'
    """
    zeroloss, oneloss = get_binary_losses(preds, labels)
    dilation = zeroloss / oneloss
    return dilation


def balanced_dilate(preds, labels, verbose=False):
    """
    Dilate either the error at label=0, or label=1 , in order to balance them.
    """
    dilation = get_binary_dilation(preds, labels)           # Ratio between both kinds of loss
    dilation = torch.unsqueeze(dilation, 0).detach()    # Must be detached!
    mask     = dilation * labels + torch.ones_like(labels)  # All 1.0 except where vectors are being altered
    
    if verbose:
        print("DILATION:", dilation)
        print("MASK:", mask.cpu().data.numpy().tolist()[100:200])

    preds  = mask * preds
    labels = mask * labels
    
    return preds, labels


def binarize(X, thresh=0.5, options={}):
    """
    Binarize a tensor without losing the ability to backpropagate
    """
    if options.get('reverse'):
        X = X.masked_fill((X<thresh), 1).masked_fill((X>=thresh), 0)
    else:
        X = X.masked_fill((X<thresh), 0).masked_fill((X>=thresh), 1)
        
    if options.get('detach'):
        X = X.detach()
        
    return X


def F1(preds, labels):
    """
    Using only pytorch, compute an F1 on two tensors of binary values [0,1]
    """
    predP    = binarize(preds)
    predN    = binarize(preds, options={'reverse':True})
    labelP   = binarize(labels)
    labelN   = binarize(labels, options={'reverse':True})
    
    TP       = torch.sum( torch.sum(labelP * predP) )
    FP       = torch.sum( torch.sum(labelN * predP) )
    TN       = torch.sum( torch.sum(labelN * predN) )
    FN       = torch.sum( torch.sum(labelP * predN) )

    F1 = (TORCH_TWO*TP) / (2*TP + FN + FP)

    return F1, TP, TN, FP, FN
    

def print_complete(T, label='T'):
    """
    For some tensor 'T', print values
    """
    print(label, ':', T.cpu().detach().numpy().tolist())


def print_100(T, label='T'):
    """
    For some tensor 'T', print first 100 values
    """
    print(label, ':', T.cpu().detach().numpy().tolist()[:100])


def squashed_leaky_relu(x):
    """
    Leaky relu up to e.  After that is squashed by log(x)
    """
    # Final activation should be like sigmoid if possible, but can't be susceptible to vanishing gradients
    x = LEAKY_RELU(x)

    # Masks thresholded at e
    hi_mask = binarize(x, thresh=TORCH_E, options={'reverse':False}).detach()
    lo_mask = binarize(x, thresh=TORCH_E, options={'reverse':True}).detach()

    # Get Low and higher log-altered version of x
    lo_x  = lo_mask * x
    hi_x  = hi_mask * x
    log_x = torch.log(hi_x)
    altered_x = log_x + TORCH_E - TORCH_ONE   # For values > 2.7187, take the log, which brings it back down to 1.0, so for continuity's sake, bring it back up to 2.7187 by adding e - 1.

    # Compile x where only higher values have been log-altered
    x = lo_x + altered_x

    return x


def has_improper_values(T):
    """
    Whether a tensor has NaN, Inf, or -Inf
    """
    if torch.sum(torch.isnan(T)) > 0:
        return True

    if torch.sum(T) in [INF, negINF]:
        return True

    return False


def tensor_to_float(T):
    T = T.cpu().data.numpy().tolist()
    return T

        
##############################################################################################
# OBJECTS

class PearsonLoss(TORCH_LOSS):
    """
    Creates a criterion that measures the Pearson Correlation Coefficient between labels and regressed output.

    The sum operation still operates over all the elements, and divides by the batch size.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.PearsonLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(PearsonLoss, self).__init__(size_average, reduce, reduction)
        self.L1 = nn.L1Loss()

 
    def pearson_coeff(self, X, Y):
        """
        The Pearson correlation coefficient is a measure of the linear correlation between two variables.

        PCC = cov(X,Y)/(stdev_X * stdev_Y)
        """
        vx   = X - torch.mean(X)
        vy   = Y - torch.mean(Y)
        pcc = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

        return pcc
       
        
    def loss(self, X, Y):
        """
        Loss function based on the Pearson Correlation Coefficient
        """
        return 1.0 - self.pearson_coeff(X, Y)


    def forward(self, input, target):
        pl = self.loss(input, target)
        return pl
        # l1 = self.L1(input, target)
        # return l1
        # return max(pl, l1)

    
class SkewedL1Loss(TORCH_LOSS):
    """
    Creates a criterion that measures the L1 Loss but greatly increases loss with respect to the 0 value

    The sum operation still operates over all the elements, and divides by the batch size.

    The division by `n` can be avoided if one sets the constructor argument
    `size_average=False`.

    Args:
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If reduce is ``False``, then
          :math:`(N, *)`, same shape as the input

    Examples::

        >>> loss = nn.PearsonLoss()
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, skew=100, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(SkewedL1Loss, self).__init__(size_average, reduce, reduction)
        self.skew = skew
        self.L1 = nn.L1Loss()

 
    def loss(self, X, Y):
        """
        Loss function based on L1 but magnifying the difference when y=0
        """
        c = self.skew * (1 - Y)
        L = (1 + c) * self.L1(X, Y)
        return torch.sum(L)
        

    def forward(self, input, target):
        pl = self.loss(input, target)
        return pl
        # l1 = self.L1(input, target)
        # return l1
        # return max(pl, l1)

    
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for PyTorch: pytorch_utils.py"})
    parser.add_argument('--rate_by_epoch', help='Test a learning rate against the rate_by_epoch function', required=False, type=float, default=0)
    args = parser.parse_args()

    if args.rate_by_epoch:
        orig = args.rate_by_epoch
        for i in range(50):
            epoch = i + 1
            output = learning_rate_by_epoch(epoch, args.rate_by_epoch)
            print("E %d  Lr: %0.8f"% (epoch, output))
        

################################################################################
################################################################################
