""" utils.py

Helper functions

"""
import os, sys, re
import traceback
import json
import pickle
import inspect
import requests
import argparse
import csv

import numpy as np
import scipy
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit

# If using spaCy for NLP
import spacy
try:
    spacy_nlp = spacy.load('en_core_web_lg')    # download separately: https://spacy.io/models/
except:
    pass

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
    
    # Treat <options> as DSObject
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

    parser.add_argument('--debug',           help='Debug Mode', required=False, action='store_true')
    parser.add_argument('--verbose',         help='Verbose mode', required=False, action='store_true')
    parser.add_argument('--test',            help='Run a test (if a model, test on the input dataset)', required=False, action='store_true')

    # Argument-taking flags
    parser.add_argument('--dir',             help='A folder having files to be read', required=False, nargs='?', action='append', )
    parser.add_argument('--file',            help='A file to be read', required=False, nargs='?', action='append', )
    parser.add_argument('--str',             help='A string to be read', required=False, nargs='?', action='append', )
    parser.add_argument('--df',              help='Panda Dataframe CSV to be read', required=False, nargs='?', action='append', )

    # Argument-taking flags (single-use)
    parser.add_argument('--output_dir',      help='Directory to save the output', required=False, type=str)
    
    return parser


def argparser_classifier(options={}):
    """
    argparser, but specific to ML

    This function should be called in the __main__ of a script.

    """
    parser = argparser(options)
    
    parser.add_argument('--load_model',      help='Load a pretrained model from disk', required=False, action='store_true')
    parser.add_argument('--load_dataset',    help='Load a prebuilt vectorizer from disk', required=False, action='store_true')
    parser.add_argument('--train',           help='Train a model on the data in the input file', required=False, action='store_true')
    parser.add_argument('--train_and_test',  help='Separate data into train/test sets, the train, then test', required=False, action='store_true')
    parser.add_argument('--classify',        help='Use the model to classify the data in the input file, the train', required=False, action='store_true')

    # Argument-taking flags (single-use)
    parser.add_argument('--model',           help='File to save the model to', required=False, type=str)
    parser.add_argument('--model_dir',       help='Directory to save the model in', required=False, type=str)
    parser.add_argument('--epochs',          help='Number of epochs for training', required=False, type=int)
    parser.add_argument('--batch_size',      help='Size of data for each epoch', required=False, type=int)
    parser.add_argument('--weights',         help='A weights file to load', required=False, type=str)

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

    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback_details = {}
    exception = None
    if isTrue(options, 'exception'):
        exception = options['exception']

    if isTrue(options, 'error'):
        sys.stderr.write('\nERROR: %s\n'% options['error'])
        
    try:
        traceback_details = {
            'filename': exc_traceback.tb_frame.f_code.co_filename,
            'lineno'  : exc_traceback.tb_lineno,
            'name'    : exc_traceback.tb_frame.f_code.co_name,
            'type'    : exc_type.__name__,
            'message' : exc_value.message, # or see traceback._some_str()
            }
        sys.stderr.write("\nDEBUG w/Trace (Line %d) from file %s:\n"% (traceback_details['lineno'], traceback_details['filename']))

        # Handle exception if it exists
        if exception is not None:
            sys.stderr.write("\tException: ["+ str(exception) +"]\n")
            sys.stderr.write("\t     type: [%s]\n"% traceback_details['type'])
            if traceback_details['message'] != str(exception):
                sys.stderr.write("\t  message: [%s]\n"% traceback_details['message'])
            tb = traceback.format_exc().rstrip()
            if tb != 'None':
                sys.stderr.write("\t traceback: [%s]\n"% tb)

    except:   # If traceback fails
        
        callerframerecord = inspect.stack()[1]    # 0 represents this line
                                                  # 1 represents line at caller
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        # func = info.function
        file = os.path.basename(info.filename)
        line = info.lineno
        sys.stderr.write("\nDEBUG (Line %d) from file %s:\n"% (line, info.filename))
        
        # Handle exception if it exists
        if exception is not None:
            sys.stderr.write( "\tException: [%s]\n"% str(exception))
            sys.stderr.write( "\t     type: [%s]\n"% str(type(exception)))

    # Print vars to STDERR if present
    if len(vars) > 0:
        if isinstance(vars, str):
            sys.stderr.write('\tVAR    %s\n' % vars)
        else:
            for v in vars:
                sys.stderr.write('\tVAR    %s\n'% str(v))
            sys.stderr.write('\n')

    if isTrue(options, 'exit'):
        exit(1)
    if isTrue(options, 'warn'):
        return
    if exception:
        raise exception


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
    final = []
    if isVerbose(options):
        sys.stderr.write("Reading file '%s' ...\n"% file)

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


def is_Model(thing):
    """
    Determine if an object is a subclass of gmutils.model.Model

    """
    for base in thing.__class__.__bases__:
        if base.__name__ == 'Model':
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
    options['joblib'] = True
    
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
            directory = thing.get('default_dir')     # assumes 'thing' is a subclass of object>Object
    except: pass
            
    # Serialize a Model
    if is_Model(thing):
        return serialize_Model(thing, directory)
        
    # Default action
    if isTrue(options, 'joblib'):
        joblib.dump(thing, file)
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
    options['verbose'] = True
    options['joblib'] = True
    
    # Deserialize a Model
    if directory is not None:
        weights_file = directory + '/trained_model.h5'
        if os.path.isfile(weights_file):
            return deserialize_Model(directory, options)
        
    if isVerbose(options):
        sys.stderr.write("Deserializing %s ...\n"% file)

    if isTrue(options, 'joblib'):
        thing = joblib.load(file)
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
        total_i = num_lines_in_file(file)  # Assumes the input must be a file
        
    lastDone = 0.0
    print ("\tLines to read:", total_i)
    sys.stderr.write("      ")
    i = 0
    _monitor = total_i, i, lastDone
    return _monitor

        
def monitor(_monitor):
    """
    To monitor progress on the command line.  See monitor_setup() above.

    """
    total_i, i, lastDone = _monitor
    i += 1
    done = 100.*float(i)/float(total_i)
    if done < 100.0  and  done - lastDone > 0.1:
        sys.stderr.write("\b\b\b\b\b\b")
        sys.stderr.write("%04.1f%% "% done)
        lastDone = done

    _monitor = total_i, i, lastDone
    return _monitor


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

    
################################################################################
##   MAIN   ##

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "utils.py"})
        #  --  Tool-specific command-line args may be added here
        args = parser.parse_args()   # Get inputs and options

        if args.file:   # Can be used for various one-off needs
            for file in args.file:
                for line in read_file(file):
                    print('Do something with Line:', line)
        else:
            print(__doc__)

    except Exception as e:
        print(__doc__)

        
################################################################################
