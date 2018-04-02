""" utils.py

Helper functions

"""
import os, sys, re
import traceback
import json
import gzip
import zipfile
import pickle
from sklearn.externals import joblib
import dill
import inspect
import requests
import argparse
import csv
import math
import numpy as np
import scipy
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from scipy import spatial

################################################################################
# FUNCTIONS

def isTrue(options, key):
    """
    To enable shorter code for determining if an option is set

    """
    # Treat <options> as dict
    try:
        if options.has_key(key):
            return options[key]
        else:
            return False
    except:
        pass

    # Treat <options> as object
    try:
        out = getattr(options, key)
        if out:
            return out
        return False
    except:
        pass
    
    # Treat <options> as Object
    try:
        out = options.get(key)
        if out:
            return out
        return False
    except:
        pass
    
    return False

        
def isVerbose(options):
    """
    To enable shorter code for determining if an option is set

    """
    return isTrue(options, 'verbose')
    

def argparser(options={}):
    """
    Set some standard options for CLI that should apply to various tools

    This function should be called in the __main__ of a script.

    'args' is a Python Namespace object.  Its attributes must exist in order to be used by conditionals.  Most are defaulted to False (see below)
    This same object can be used to provide access to 'options' from other functions and objects.

    """
    desc = None
    if isTrue(options, 'desc'):
        desc = options['desc']
    
    parser = argparse.ArgumentParser(description=desc)

    # Boolean flags
    parser.add_argument('--debug',            help='Debug Mode', required=False, action='store_true')
    parser.add_argument('--test',             help='Run a test', required=False, action='store_true')
    parser.add_argument('--verbose',          help='Verbose mode', required=False, action='store_true')
    parser.add_argument('--normalize',        help='Normalize input text', required=False, action='store_true')

    # Argument-taking flags
    parser.add_argument('--df',               help='Panda Dataframe CSV to be read', required=False, nargs='?', action='append')
    parser.add_argument('--dir',              help='A folder having files to be read', required=False, nargs='?', action='append')
    parser.add_argument('--file',             help='A file to be read', required=False, nargs='?', action='append')
    parser.add_argument('--str',              help='A string to be read', required=False, nargs='?', action='append')

    # Argument-taking flags (single-use)
    parser.add_argument('--bucket_id',        help='Google bucket ID', type=str, required=False, default=None, action='append')
    parser.add_argument('--output_dir',       help='Directory to save the output', required=False, type=str)
    parser.add_argument('--host',             help='Host/IP address', required=False, type=str)
    parser.add_argument('--port',             help='Port number', required=False, type=int)
    parser.add_argument('--skip',             help='Skip (can be for skipping ahead through long files or processes)', required=False, type=str)
    
    return parser


def argparser_ml(options={}):
    """
    argparser, but specific to ML

    This function should be called in the __main__ of a script.

    """
    parser = argparser(options)
    
    # Boolean flags
    parser.add_argument('--balance_by_copies', help='Balance the data classes (training only) by simply copying some samples', required=False, action='store_true')
    parser.add_argument('--classify',         help='Use the model to classify the data in the input file, the train', required=False, action='store_true')
    parser.add_argument('--eval',             help='Separate data into train/eval sets, then evaluate a trained model', required=False, action='store_true')
    parser.add_argument('--load_dataset',     help='Load a prebuilt vectorizer from disk', required=False, action='store_true')
    parser.add_argument('--load_model',       help='Load a pretrained model from disk', required=False, action='store_true')
    parser.add_argument('--train',            help='Train a model on the data in the input file', required=False, action='store_true')
    parser.add_argument('--train_and_eval',   help='Separate data into train/eval sets, then train, then evaluate the trained model', required=False, action='store_true')

    # Argument-taking flags (single-use)
    parser.add_argument('--batch_size',       help='Size of data for each epoch', required=False, type=int)
    parser.add_argument('--data_dir',         help='Directory where data is stored', required=False, type=str)
    parser.add_argument('--dataset_file',     help='Load a specific dataset file', required=False, type=str)
    parser.add_argument('--epochs',           help='Number of epochs for training', required=False, type=int)
    parser.add_argument('--eval_file',        help='Evaluation files local or GCS', required=False, type=str)
    parser.add_argument('--eval_dir',         help='Directory where eval data is stored', required=False, type=str)
    parser.add_argument('--label_column',     help='Output label for which to train', type=str, required=False)
    parser.add_argument('--learning_rate',    help='Learning rate for SGD', type=float, default=0.003)
    parser.add_argument('--model',            help='File to save the model to', required=False, type=str)
    parser.add_argument('--model_dir',        help='Directory to save the model in', required=False, type=str)
    parser.add_argument('--model_file',       help='Load a specific model file', required=False, type=str)
    parser.add_argument('--thresh',           help='Threshold for some output label operations such as binarization', type=float, required=False)
    parser.add_argument('--train_file',       help='Training files local or GCS', required=False, type=str)
    parser.add_argument('--train_dir',        help='Directory where training data is stored', required=False, type=str)
    parser.add_argument('--steps_per_epoch',  help='Steps per epoch', required=False, type=int)
    parser.add_argument('--weights',          help='A weights file to load', required=False, type=str)

    return parser

    
def err(vars=[], options={}):
    """
    Prints var values and/or filename and debug info STDERR.  This is easier than writing comparable print statements,
    and there are times when the Python interpreter doesn't output all of this information.

    Paramters
    ---------
    vars : list or str
        a list of values to be printed

    Options
    -------
    verbose : boolean

    exception : Exception (optional)

    ex : create and Exception

    GM_LEVEL (env var)
        Lowest value allows all errors and warnings to print

    """
    verbose = False
    if verbose:  sys.stderr.write("err 162\n")
    
    # Information about the urgency of this call
    call_level = options.get('level')
    if call_level is None:  call_level = 2    # default is 2

    os_level = 0
    try:
        os_level = int(os.environ['GM_LEVEL'])
    except:  pass
        
    if call_level < os_level:
        options['silent'] = True

    if verbose:  sys.stderr.write("err 176\n")
        
    # Gather frame
    callerframerecord = inspect.stack()[1]    # 0 represents this line
                                              # 1 represents line at caller
    if verbose:  sys.stderr.write("err 181\n")
        
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    file = os.path.basename(info.filename)
    
    if verbose:  sys.stderr.write("err 187\n")
        
    line = info.lineno
    if not isTrue(options, 'silent'):
        sys.stderr.write("\nDEBUG (Line %d) from file %s:\n"% (line, info.filename))

    if verbose:  sys.stderr.write("err 193\n")
        
    # Parse exception
    exception = options.get('exception')
    if exception is None:
        if options.get('ex'):
            exception = ValueError(options.get('ex'))
            
    if isTrue(options, 'exception'):
        exception = options['exception']
        if not isTrue(options, 'silent'):
            for arg in exception.args:
                sys.stderr.write("ERROR: {}\n".format(arg))
            sys.stderr.write("\n\t"+ str(sys.exc_info()[0]) +"\n")

    if verbose:  sys.stderr.write("err 208\n")
            
    # Print vars to STDERR if present
    if len(vars) > 0:
        if verbose:  sys.stderr.write("err 212\n")
        if isinstance(vars, str):
            if verbose:  sys.stderr.write("err 214\n")
            sys.stderr.write('\tVAR  |%s|\n' % vars)
        else:
            if verbose:  sys.stderr.write("err 217: " + str(vars) + "\n")
            for v in vars:
                if verbose:  sys.stderr.write("err 219: " + str(v) + "\n")
                sys.stderr.write('\tVAR  |%s|  %s\n'% (str(v), str(type(v))) )
            sys.stderr.write('\n')

    if verbose:  sys.stderr.write("err 223\n")
            
    # Conditional return
    if isTrue(options, 'exit'):
        exit(1)
    if isTrue(options, 'warning')  or  isTrue(options, 'silent')  or  call_level < 2:
        if not isTrue(options, 'silent'):
            for arg in exception.args:
                sys.stderr.write("ERROR: {}\n".format(arg))
            sys.stderr.write("\n\t"+ str(sys.exc_info()[0]) +"\n")
        return
    if exception:
        raise exception


def read_zipfile(file, options={}):
    """
    Read text from 'file', where the file has been zipped up and has a name which is 'file' minus the extension

    Parameters
    ----------
    file : str

    Options
    -------
    'one str' : if True, will return a single str (By default, each line is a separate str)

    Returns
    -------
    str or array of str
    """
    zipfile = Zipfile(file)
    innerfile = re.sub(r'\.zip$', '', file)
    innerfile = re.sub(r'\.gz$', '', innerfile)
    
    final = []
    with open(zipfile.open(innerfile)) as resource:
        for line in resource:
            final.append(line.rstrip())

    if isTrue(options, 'one str'):
        return "\n".join(final)

    return final
    

def read_file(file, options={}):
    """
    Read text from 'file'

    Parameters
    ----------
    file : str

    Options
    -------
    'one str' : if True, will return a single str (By default, each line is a separate str)

    Returns
    -------
    str or array of str
    """
    if isVerbose(options):
        sys.stderr.write("Reading file '%s' ...\n"% file)

    if re.search(r'\.zip$', file)  or  re.search(r'\.zip$', file):
        return read_zipfile(file, options=options)
        
    final = []
    with open(file) as resource:
        for line in resource:
            final.append(line.rstrip())

    if isTrue(options, 'one str'):
        return "\n".join(final)

    return final


def iter_file(file, options=None):
    """
    Create an iterator for 'file' to read it line by line

    Parameters
    ----------
    file : str

    Returns
    -------
    iterator

    """
    if isVerbose(options):
        sys.stderr.write("Creating iterator for file '%s' ...\n"% file)

    FH = open(file, 'r')
    return iter(FH)


def write_file(file, content, options={}):
    """
    Write 'content' to 'file'

    Parameters
    ----------
    content : (probably) str or str[]

    file : str

    Returns
    -------
    str or array of str
    """
    if isVerbose(options):
        sys.stderr.write("Writing to %s ...\n"% file)

    with open(file, 'w') as outfile:

        if isinstance(content, str):
            outfile.write(content)

        elif isinstance(content, list):
            for line in content:
                if re.search('\n$', line):
                    outfile.write(line)
                else:
                    outfile.write(line + '\n')
        else:
            err([type(content), 'ERROR: Unknown type'])


def read_dir(path, options={}):
    """
    Get list of all elements stored in a given directory

    """
    verbose = False
    if verbose:
        err([path])
    (dirpath, folders, files) = next(os.walk(path))

    # Files having a certain file extension (suffix)
    if isTrue(options, 'suffix'):
        filtered = []
        for file in files:
            if re.search(options.suffix +'$', file):
                filtered.append(file)
        files = filtered

    # Add full path to files
    if isTrue(options, 'fullpath'):
        filtered = []
        for file in files:
            filtered.append(dirpath +'/'+ file)
        files = filtered

    if isTrue(options, 'files'):
        return files
    elif isTrue(options, 'folders'):
        return folders
    out = folders
    out.extend(files)

    return out


def serialize_and_save_conceptnet_vectorfile(vectorfile, pklfile):
    """
    Read vocab and embedding from file and serialize to disk

    """
    def preprocess(word, vector):
        """
        Carefully process and discard some words from this embedding
        """
        word = ascii_fold(word)
        word = normalize(word)
        if not re.search(r'^[a-z_\'’]*$', word):
            word = False
            vector = None
        return word, vector

    vectors = read_conceptnet_vectorfile(vectorfile, {'langs':['en'], 'preprocess':preprocess})
    print("Serializing %d vectors to the file: %s ..."% (len(vectors), pklfile))
    serialize(vectors, pklfile)


def read_conceptnet_vectorfile(filename, options={}):
    """
    Read a file formatted like a ConceptNet Numberbatch file.  These vectors are simple arrays of float.
    This function is only necessary because of an error related to either h5py or how this file was created.

    Parmmeters
    ----------
    filename : str

    Options
    -------
    preprocess : func
        str,vector -> str,vector, or False,None
        A function that alters some entries, leaves others untouched, and removes some

    Returns
    -------
    dict of dict of dict (str:vector)
        vocab : <languange code> : <word> : <vector>

    """
    verbose = False
    vocab = {}
    repeats = {}   # will be used at the end to compute the average vector for any collisions
    langs = options.get('langs')

    # Iterate through the file
    for line in read_file(filename):

        # Parse each line
        if not re.search(r'^/', line): continue
        wordpath, *vector = line.split()
        try:
            none, kind, lang, word, pos = wordpath.split('/')
        except:
            pos = None
            none, kind, lang, word = wordpath.split('/')

        # Ignore some lines, check for errors
        if kind != 'c':
            err([],{'ex':"Unexpected concept type: %s"% kind})  # The default file only has 'c' entries (concepts)
        if langs is not None:
            if lang not in langs: continue
        if pos is not None:
            err([],{'ex':"Unexpected POS: %s for %s"% (pos, word)})  # The default file only has 'c' entries (concepts)
            
        vector = np.array(list(map(float, vector)))   # convert to numpy array of floats
        
        # Preprocessing.  After this step collisions sometimes occur
        if options.get('preprocess'):
            word, vector = options.get('preprocess')(word, vector)
            if word == False:
                continue
            
        if re.search(r'^[a-z_\'’]*$', word):
            pass
        else:
            if verbose: print('IGNORE:', word)
            continue

        # Structure outgoing data into a dict of dicts.  Where necessary, handle collisions by summing the vectors
        if lang in vocab:
            if word in vocab[lang]:
                if word in repeats[lang]:
                    repeats[lang][word].append(vector)
                else:
                    repeats[lang][word] = [vector]
            else:
                vocab[lang][word] = vector
        else:
            vocab[lang] = { word:vector }
            repeats[lang] = {}

    # For each repeated entry, compute an averaged vector
    for lang in repeats.keys():
        for word in repeats[lang].keys():
            vocab[lang][word] = vector_average([vocab[lang][word]] + repeats[lang][word])
            
    return vocab

    
def is_KerasModel(thing):
    """
    Determine if an object is a subclass of gmutils.model.Model

    """
    for base in thing.__class__.__bases__:
        if base.__name__ == 'KerasModel':
            return True
    return False
    

def serialize(thing, file=None, directory=None, options={}):
    """
    Serialize an object and save it to disk

    Parameters
    ----------
    thing : subclass of gmutils.objects.Object

    file : str

    directory : str
    """
    # Informative STDERR output
    if isVerbose(options):
        thingType = re.sub(r"^.*'(.*)'.*$", r"\1", (str(type(thing))))
        thingType = re.sub(r"__main__\.", "", thingType)
        sys.stderr.write("Saving %s to %s ...\n"% ( thingType, file))

    # Determine location
    try:
        if file is None:
            file  = thing.get('default_file')        # assumes 'thing' is a subclass of object>Object
        if directory is None:
            directory = thing.get('default_dir')
    except: pass
            
    # Serialize a Keras Model
    if is_KerasModel(thing):
        return serialize_KerasModel(thing, directory)
        
    # Default action
    if directory is None:
        filepath = file
    else:
        filepath = directory +'/'+ file
        
    if isTrue(options, 'joblib'):
        joblib.dump(thing, file)
    elif isTrue(options, 'dill'):
        with open(file,'wb') as FH:
            dill.dump(thing, FH)
    else:
        with open(file,'wb') as FH:
            pickle.dump(thing, FH)


def deserialize(file=None, directory=None, options={}):
    """
    De-Serialize an object from disk

    Parameters
    ----------
    file : str

    directory : str

    """
    # options['verbose'] = True
    
    # Deserialize a Model
    if directory is not None:
        weights_file = directory + '/trained_model.h5'
        if os.path.isfile(weights_file):
            return deserialize_KerasModel(directory, options)
        
    if isVerbose(options):
        sys.stderr.write("Deserializing %s ...\n"% file)

    if isTrue(options, 'joblib'):
        thing = joblib.load(file)
        return thing
    elif isTrue(options, 'dill'):
        with open(file,'rb') as FH:
            try:
                thing = dill.load(FH)
                return thing
            except:
                return thing
    else:
        with open(file,'rb') as FH:
            thing = pickle.load(FH)
            return thing


def serialize_Model(thing, directory, options={}):
    """
    Serialize a Model and save it to <directory>

    Parameters
    ----------
    thing : subclass of gmutils.model.Model

    directory : str

    """
    # If the model has an underlying Keras Model, save the weights
    thing.model.save_weights(directory + '/trained_model.h5')

    
def deserialize_Model(directory, options={}):
    """
    Deserialize a Model and save it to <directory>

    Parameters
    ----------
    directory : str

    """
    weights_file = directory + '/trained_model.h5'
    model = options['constructor'](options)
    model.load_weights(weights_file)
    
    
def close_enough(a, b):
    if abs(a-b) < 0.00001:
        return True
    return False


def file_exists(path):
    """
    Determine if a path is a file and it exists
    """
    if os.path.isfile(path):
        return True
    return False


def dir_exists(path):
    """
    Determine if a path is a file and it exists
    """
    if os.path.isdir(path):
        return True
    return False


def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    except Exception as e:
        err([traceback.format_exception(*sys.exc_info())])


def mkdirs(paths):
    for path in paths:
        mkdir(path)


def num_lines_in_file(file):
    """
    Get the number of lines in a file
    """
    for line in os.popen("wc -l %s"% file):
        if re.search('^\s*\d', line):
            lno = re.sub(r'^\s*(\d+)\s.*$', r'\1', line)
            return int(lno)
    return -1


def num_from_filename(file):
    """
    In such cases, the number of paragraphs will be included at the end of the filename, like "file_n25.json"
    """
    try:
        n = re.sub(r'^.*_n(\d+)\.json$', r'\1', file)
        n = int(n)
    except:
        n = 1
    return n
    

def monitor_setup(file, total_i=None):
    """
    To setup monitoring for the progess of a loop.  Use in conjunction with monitor()

    Examples:

    _monitor = monitor_setup(None, len(arr))
    for thing in arr:
        _monitor = monitor(_monitor)

    _monitor = monitor_setup(filename)
    for line in read_file(filename):
        _monitor = monitor(_monitor)

    """
    if total_i is None:
        if re.search(r'_n\d', file):
            total_i = num_from_filename(file)
        else:
            total_i = num_lines_in_file(file)  # Assumes the input must be a file
        
    lastDone = 0.0
    sys.stderr.write("\tLines to read: %d\n"% total_i)
    sys.stderr.write("      ")
    i = 0
    _monitor = total_i, i, lastDone
    return _monitor

        
def monitor(_monitor, skip=None, options={}):
    """
    To monitor progress on the command line.  See monitor_setup() above.

    """
    total_i, i, lastDone = _monitor
    skip_state = False   # default: only used if 'skip' is set
    progress = None
    i += 1

    done = 100.*float(i)/float(total_i)
    if done < 100.0  and  done - lastDone > 0.005:
        if options.get('progress_str'):
            progress = "%04.2f%% "% done
        if not options.get('silent'):
            sys.stderr.write("\b\b\b\b\b\b\b")
            sys.stderr.write("%04.2f%% "% done)
            sys.stderr.flush()
        lastDone = done
        
    _monitor = total_i, i, lastDone

    if skip is None:
        if progress is None:
            return _monitor
        else:
            return _monitor, progress

    elif float(skip) > done:
        skip_state = True

    if progress is None:
        return _monitor, skip_state
    else:
        return _monitor, skip_state, progress


def split_data(X, Y, ratios):
    """
    Split dataset, keeping rows associated between X, Y 

    Parameters
    ----------
    data : DataFrame containing both X and Y

    X : DataFrame or list of rows which can be used to generate a DataFrame

    Y : Series or list of supervised outputs

    ratios : list of float
        Proportion of the data to apportion to each subset

    Returns
    -------
    X0 : numpy array
    Y0 : list
    X1 : numpy array
    Y1 : list
    etc.

    """
    out = []  # output array

    # Load everything together in one dataframe
    df = pd.DataFrame(X)
    if '_Y_' in df:
        err(['ERROR: input data already has _Y_'])
        exit()
    df['_Y_'] = pd.Series(Y)
    
    shuffled = df.sample(frac=1)       # Shuffle the data
    n = shuffled.shape[0]
    last_i = 0

    # Put the requested number of lines in each partition
    for ratio in ratios:
        i = last_i + int(ratio * n)  # index of the end of this partition
        partial = df[last_i:i]       # dataframe having only this partition
        this_X = partial.loc[:, partial.columns != '_Y_'].as_matrix()
        this_Y = partial['_Y_'].tolist()
        out.append( (this_X, this_Y) )
        last_i = i

    return out
    

def is_jsonable(x):
    """
    Determine if an object can be converted to json using the json module.

    """
    try:
        json.dumps(x)
        return True
    except:
        return False


def set_missing_attributes(namespace, attributes=None):
    """
    Set some missing options using a dict of defaults.
    Some options may have been missing because they either weren't serializable or simply weren't specified.

    Parameters
    ----------
    namespace : Namespace

    attributes : dict

    """
    if attributes is not None:
        for param in attributes.keys():
            try:
                if not getattr(namespace, param):
                    setattr(namespace, param, attributes[param])
            except:
                setattr(namespace, param, attributes[param])


def override_attributes(namespace, attributes=None):
    """
    Set some missing options using a dict of defaults.  Override any existing values.

    Parameters
    ----------
    namespace : Namespace

    attributes : dict

    """
    if attributes is not None:
        for param in attributes.keys():
            setattr(namespace, param, attributes[param])


def vector_average(vectors):
    """
    Return the vector average of a list of floating-point vectors

    Parameters
    ----------
    vectors : array of array of float, all of the same length

    Returns
    -------
    same-length vector with averaged values

    """
    arr = np.array(vectors)
    arr = np.mean(arr, axis=0)
    return arr


def start_with_same_word(A, B):
    """
    Split strings A and B into words.  See if they begin with matching words.
    """
    a_words = A.split(' ')
    b_words = B.split(' ')
    if a_words[0] == b_words[0]:
        return True
    return False
    

def pandasize(X):
    """
    Convert some incoming data to a pandas DataFrame or Series, depending on its dimensionality

    Parameters
    ----------
    X : list, array, or numpy Array

    Returns
    -------
    pandas Series or DataFrame
    """
    make_series = False   # Assume DataFrame until otherwise indicated
    
    if isinstance(X, pd.Series):
        return X
    elif isinstance(X, pd.DataFrame):
        return X
        
    elif isinstance(X, list):
        make_series = True
    
    elif isinstance(X, np.ndarray):

        if len(X.shape) == 1:
            make_series = True

        if len(X.shape) == 2:
            if X.shape[0] == 1:
                make_series = True
                X = X[0]

    if make_series:
        X = pd.Series(X)

    else:
        X = pd.DataFrame(X)

    return X


def cosine_similarity(A, B):
    """
    The cosine similarity between two vectors, A and B

    Parameters
    ----------
    A, B : lists of equal length

    Returns
    -------
    float
        0.0 : least similar
        1.0 : most similar

    """
    if A is None or B is None:
        return 0.0   # more useful that None is many situations
    
    distance = spatial.distance.cosine(A, B)
    if math.isnan(distance):
        return 0.0
    similarity = 1.0 - distance
    
    return similarity


def binary_distance(a, b):
    """
    Same as the Levenshtein distance between two strings of 1s and 0s, but computed using binary operations.
    """
    if isinstance(a, str):
        a = int(a, 2)
    if isinstance(b, str):
        b = int(b, 2)
    c = bin(a ^ b)
    
    return c.count('1')


def json_dump_gz(filepath, data):
    """
    Dump 'data' into a JSON str and save it in a .gz file
    """
    json_str = json.dumps(data) + "\n"
    json_bytes = json_str.encode('utf-8')
    with gzip.GzipFile(filepath, 'w') as fout:
        fout.write(json_bytes)


def json_load_gz(filepath):
    with gzip.GzipFile(filepath, 'r') as fin:
        json_bytes = fin.read()
    json_str = json_bytes.decode('utf-8')
    data = json.loads(json_str)
    
    return data


def deepcopy_list(X):
    """
    Deepcopy the list X (Python deepcopy fails for many object types)
    """
    out = []
    for x in X:
        out.append(x)
    return out
    

################################################################################
# MAIN

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "utils.py"})
        parser.add_argument('--pklfile', help='Target pickle file.', required=False, type=str)
        parser.add_argument('--binary_dist', help='Binary distance between two str (comma delim)', required=False, type=str)
        args = parser.parse_args()   # Get inputs and options

        if args.file:   # Can be used for various one-off needs

            if args.pklfile:   # Read and save a vector file
                serialize_and_save_conceptnet_vectorfile(args.file[0], args.pklfile)

            else:
                for file in args.file:
                    for line in read_file(file):
                        print('Do something with Line:', line)

        elif args.binary_dist:
            a, b = args.binary_dist.split(',')
            a = int(a)
            b = int(b)
            print(binary_distance(a, b))
                        
        elif args.test:
            a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0]
            #b = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
            b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            print(cosine_distance(a,b))
                    
        else:
            print(__doc__)

    except Exception as e:
        print(__doc__)

        
################################################################################
################################################################################
