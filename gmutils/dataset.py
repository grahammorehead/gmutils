""" dataset.py

Code and objects to manage datasets for training models

"""
import os, sys, re
import numpy as np

from gmutils.utils import err, argparser, read_dir, read_file
from gmutils.objects import Object

try:
    from sklearn.model_selection import train_test_split
except Exception as e: err([], {'exception':e, 'level':0})
try:
    import pandas as pd
except Exception as e: err([], {'exception':e, 'level':0})
    
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

    Attributes (some optional)
    ----------
    x_train : pandas DataFrame
    y_train : pandas Series

    x_test : pandas DataFrame
    y_test : pandas Series

    x_validation : pandas DataFrame
    y_validation : pandas Series

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

        if self.get('balance_by_copies'):   # only affects training data
            self.balance_by_copies()
        

    def explicit_load(self, x_train, x_test, y_train, y_test):
        """
        Eplicitly load sets
        """
        self.x_train  = x_train
        self.x_test   = x_test
        self.y_train  = y_train
        self.y_test   = y_test
            
        self.print_set_sizes()

        if self.get('balance_by_copies'):   # only affects training data
            self.balance_by_copies()

            
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
        try:
            n_train = len(self.y_train)         # Lines of input text for training
        except:
            n_train = 0
        try:
            n_test  = len(self.y_test)          # Lines of input text for testing
        except:
            n_test  = 0
            
        sys.stderr.write('\nTraining set:  %s\n'% str(n_train))
        sys.stderr.write('Test set:      %s\n'% str(n_test))

    
    def get_class_counts(self, Y):
        """
        To know how many of each class is in a given set of output labels

        Parameters
        ----------
        Y : pandas.Series
        
        Options
        -------
        thresh : float 
            Uses this object-wide option in the case that the labels are continuous

        Returns
        -------
        counts : a dict having each class name and number of occurrences
        """
        counts = {}

        # For continuous data, a threshold must have been set
        thresh = self.get('thresh')

        if thresh is not None:
            counts[0] = 0
            counts[1] = 0
            for y in Y:
                if y > thresh:
                    counts[1] += 1
                else:
                    counts[0] += 1

        else:
            vc = Y.value_counts()
            for i,v in vc.iteritems():
                counts[i] = v
                
        return counts


    def get_positive_testdata(self):
        """
        Return a subset of the test data: only that which is either positive or above <thresh> (if provided)

        Options
        -------
        thresh : float
            Object-level option for dealing with continuous labels

        Returns
        -------
        X : pandas DataFrame

        Y : pandas Series
            either floats with a thresh, or (0,1, other), or boolean

        """
        out_x = []
        out_y = []
        thresh = self.get('thresh')
        
        for i, row in self.x_test.iterrows():
            y = self.y_test.loc[i]

            if thresh is None:
                if y:
                    out_x.append(row)
                    out_y.append(y)
            
            elif y > thresh:
                out_x.append(row)
                out_y.append(y)

        X = pd.DataFrame(out_x)
        Y = pd.Series(out_y)

        return X, Y


    def iter_training_samples_by_thresh(self, aorb, thresh):
        """
        Yield training samples above or below a threshold
        """
        if aorb == 'above':
            while True:
                for i, row in self.x_test.iterrows():
                    y = self.y_test.loc[i]
                    if y > thresh:
                        yield row, y
                    
        elif aorb == 'below':        
            while True:
                for j, row in self.x_test.iterrows():
                    y = self.y_test.loc[j]
                    if y <= thresh:
                        yield row, y
                    
        else:  err([], {'ex':"Unexpected"})


    def get_combined_training_copy(self, thresh=None, aorb=None):
        """
        For the purposes of oversampling a minority set, return a deepcopy of the training data with X and Y in one DataFrame.

        If specified, use a threshold to prune some of the data.
        """
        df = self.x_train.copy()
        df['Y'] = self.y_train.copy()

        if thresh is None:
            pass
        elif aorb == 'above':
            df = df[df['Y'] > thresh]
            cc = self.get_class_counts(df['Y'])
        elif aorb == 'below':
            df = df[df['Y'] <= thresh]
            
        return df
    
            
    def get_combined_training_copy_for_class(self, cl):
        """
        For the purposes of oversampling a minority set, return a deepcopy of the training data with X and Y in one DataFrame.

        Only returns the subset with output labels matching class 'cl'

        """
        df = self.x_train.copy()
        df['Y'] = self.y_train.copy()
        df = df[df['Y']==cl]
        
        return df
    
            
    def get_training_samples_by_class(self, cl, diff):
        """
        Gather 'diff' training samples matching class 'cl'
        """
        x_new = pd.DataFrame([])
        y_new = pd.Series([])

        tdf = self.get_combined_training_copy_for_class(cl)
        
        while len(y_new) < diff:
            x_new = x_new.append(tdf.drop(['Y'], axis=1))
            y_new = y_new.append(tdf['Y'])

        if len(y_new) > diff:
            x_new = x_new[:diff]
            y_new = y_new[:diff]
            
        return x_new, y_new

    
    def get_training_samples_by_thresh(self, aorb, thresh, diff):
        """
        Gather 'diff' training samples above or below a threshold
        """
        x_new = pd.DataFrame([])
        y_new = pd.Series([])

        tdf = self.get_combined_training_copy(thresh, aorb)
        
        while len(y_new) < diff:
            x_new = x_new.append(tdf.drop(['Y'], axis=1))
            y_new = y_new.append(tdf['Y'])

        if len(y_new) > diff:
            x_new = x_new[:diff]
            y_new = y_new[:diff]
            
        return x_new, y_new

    
    def balance_by_copies(self):
        """
        If one or more classes are minority, increase their ranks by copying original set enough times to reach
        size parity with the majority class.  AKA "Minority Oversampling"

        This applies ONLY to the training set.  Test and validation or left untouched.
        """
        cc = self.get_class_counts(self.y_train)
        print("Training class counts:", cc)
        sys.stderr.write('Balancing by oversampling copies of the minority set ...\n')
        X = self.x_train
        Y = self.y_train
        x_new = pd.DataFrame([])
        y_new = pd.Series([])
        thresh = self.get('thresh')  # set IFF output labeling is continous
        max_count = max(cc.values())
        
        ### Balancing with classes
        if thresh is None:
            sizes = {}
            vc = Y.value_counts()
            max_cl = 0
            
            for cl, times in vc.iteritems():
                if cl > max_cl:
                    max_cl = cl
                    
            for cl in cc.keys():
                diff = max_count - cc[cl]
                if diff > 0:
                    xn, yn = self.get_training_samples_by_class(cl, diff)
                    x_new = x_new.append(xn)
                    y_new = y_new.append(yn)

        ### Balancing by threshold  (for continuous data)
        else:
            diff0 = max_count - cc[0]
            diff1 = max_count - cc[1]

            if diff0 > 0:
                x_new, y_new = self.get_training_samples_by_thresh('below', thresh, diff0)
            elif diff1 > 0:
                x_new, y_new = self.get_training_samples_by_thresh('above', thresh, diff1)
                    
        self.x_train = self.x_train.append(x_new)
        self.y_train = self.y_train.append(y_new)
                
        cc = self.get_class_counts(self.y_train)
        print("Training class counts:", cc)
        
            
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
    
