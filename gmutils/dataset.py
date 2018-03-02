""" dataset.py

Code and objects to manage datasets for training models

"""
import os, sys, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gmutils.utils import err, argparser, read_dir, read_file
from gmutils.objects import Object

################################################################################
# DEFAULTS

default = {
    'default_file'        : 'dataset',
}

    
################################################################################
# OBJECTS

class Dataset(Object):
    """
    A dataset object to simplify the training and use of a model

    Attributes
    ----------
    x_train : numpy ndarray
    y_train : numpy ndarray

    x_test : numpy ndarray
    y_test : numpy ndarray

    x_validation : numpy ndarray
    y_validation : numpy ndarray

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options

        """
        self.set_options(options, default)        # For more on 'self.set_options()' see object.Object

        
    def load(self, input, read_data=None):
        """
        Using a prescribed reading function, 'read_data', this function loads data into the object's attributes.

        Parameters
        ----------
        input : text file, CSV, panda DataFrame, directory, or other, depending on the read_data function

        read_data : a function which can read the specified input

        """
        if read_data is not None:
            self.x_train, self.x_test, self.y_train, self.y_test = read_data(input)
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = self.read_data(input)
            
        self.print_set_sizes()


    def read_data(self, inputs):
        """
        Default function for reading data into a Dataset.  Will accept one of:
            - file
            - list of files
            - dir

        Parameters
        ----------
        inputs : DataFrame or array thereof

        Returns
        -------
        array of DataFrame (4 of them: x_train, x_test, y_train, y_test)

        """
        if isinstance(inputs, list):
            if len(inputs) == 1:
                inputs = inputs[0]
            elif len(inputs) > 1:
                return default_read_data_files(inputs)
            else:
                err([],{'ex':"ERROR: zero-length array of inputs"})

        if isinstance(inputs, str):
            if os.path.isfile(inputs):
                return default_read_data_file(inputs)
            elif os.path.isdir(inputs):
                return default_read_data_dir(inputs)
            else:
                err([],{'ex':"ERROR: inputs neither file nor dir."})

        else:
            err([], {'ex':'Unrecognized input type: %s'% type(inputs)})

        return x_train, x_test, y_train, y_test

        
    def print_set_sizes(self):
        """
        Print info about the training and test sets
        """
        n_train = len(self.y_train)         # Lines of input text for training
        n_test  = len(self.y_test)          # Lines of input text for testing
        sys.stderr.write('Training set:  %s\n'% str(n_train))
        sys.stderr.write('Test set:      %s\n'% str(n_test))

    
    def get_class_counts(self, Y):
        """
        To know how many of each class is in the training samples

        Parameters
        ----------
        Y : pandas.Series

        Returns
        -------
        counts : a dict having each class name and number of occurrences
        """
        counts = {}
        for y in Y:
            y = int(y)
            if y in counts:
                counts[y] += 1
            else:
                counts[y] = 1
                
        return counts
            
            
################################################################################
# FUNCTIONS

def default_read_data_file(input_file):
    """
    Default function for reading data into a Dataset

    Parameters
    ----------
    input_file : str
        DataFrame saved as csv file

    Returns
    -------
    array of DataFrame (4 of them: x_train, x_test, y_train, y_test)

    """
    print('Reading DataFrame input_file file %s ...'% input_file)
    df = pd.read_csv(input_file, index_col=0)
    
    # Determine supervised label
    label = None
    if 'Y' in df.columns:
        label = 'Y'
    elif '_Y_' in df.columns:
        label = '_Y_'
    else:
        print('ERROR: no label found')
        exit()

    X = df.loc[:, df.columns != label]
    Y = df[label]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)
    
    return x_train, x_test, y_train, y_test


def default_read_data_files(input_files):
    """
    Default function for reading data files into a Dataset

    Parameters
    ----------
    input_files : array of str
        DataFrames saved as csv files

    Returns
    -------
    array of DataFrame (4 of them: x_train, x_test, y_train, y_test)

    """
    x_train  = pd.DataFrame([])
    x_test   = pd.DataFrame([])
    y_train  = pd.DataFrame([])
    y_test   = pd.DataFrame([])
    for input_file in input_files:
        x_tr, x_te, y_tr, y_te = default_read_data_file(input_file)
        x_train = pd.concat([x_train, x_tr])
        x_test = pd.concat([x_test, x_te])
        y_train = pd.concat([y_train, y_tr])
        y_test = pd.concat([y_test, y_te])
        
    return x_train, x_test, y_train, y_test

    
def default_read_data_dir(input_dir):
    """
    Default function for reading data from a directory into a Dataset

    Parameters
    ----------
    input_dir : str
        Directory where each file will be read

    Returns
    -------
    array of DataFrame (4 of them: x_train, x_test, y_train, y_test)

    """
    files = read_dir(input_dir, {'fullpath':True})
    return default_read_data_files(files)
    
    

################################################################################
##   MAIN   ##

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "Tools to manage a training set: dataset.py"})
        #  --  Tool-specific command-line args may be added here
        args = parser.parse_args()   # Get inputs and options

        print(__doc__)

    except Exception as e:
        print(__doc__)
        err([], {'exception':e, 'exit':True})

        
################################################################################
################################################################################
    
