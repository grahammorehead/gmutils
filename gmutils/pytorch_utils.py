""" pytorch_utils.py

    A set of utils to make PyTorch code simpler

"""
import sys, os, re
import random
import shutil
import inspect
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss as Loss
from torch.optim.optimizer import Optimizer
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
    

def torchtensor(X, ttype=torch.DoubleTensor, requires_grad=False):
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
            exit()
        T = ttype(X)

    T.requires_grad = requires_grad
    # print_info(T)
    return T

    
def torchvar(X, ttype=torch.DoubleTensor, requires_grad=False):
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


def torchvar_list(X, ttype=torch.DoubleTensor, requires_grad=False):
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


def var_zeros(n, ttype=torch.DoubleTensor, requires_grad=False):
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
    

def var_ones(n, ttype=torch.DoubleTensor, requires_grad=False):
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
    return lr * (0.5 ** (epoch-1))


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

    
def clear_chaff_by_loss(dirpath, MAX=100):
    """
    Get a list of all models sorted by loss.  Keep MAX, and delete the rest
    """
    verbose = False

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
    

class PearsonLoss(Loss._Loss):
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