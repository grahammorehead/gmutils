""" dataset.py

Code and objects to manage datasets for training models

"""
import os, sys, re
print('PATH:', sys.path)
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import *
from objects import Object


################################################################################
# OBJECTS

class Dataset(Object):
    """
    A dataset object to simplify the training and use of a model

    Attributes
    ----------

    data : panda dataframe
        The original data, associating input text with labels

    cols : str[]
        Column headers

    signal : str
        This is the column header of the column for supervised output, the "Y"

    train : panda dataframe
        subset of the dataframe used for training

    test : panda dataframe
        subset of the dataframe used for testing

    validate : panda dataframe
        another subset created upon request

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options

        """
        self.set_options(options)          # For more on 'self.set_options()' see objects.py for the Object class

        
    def load(self, input=None, read_data=None):
        """
        Using a prescribed reading function, 'read_data', this function loads data into the object's attributes.

        Parameters
        ----------
        input : text file, CSV, panda DataFrame, directory, or other, depending on the read_data function

        read_data : a function which can read the specified input

        """
        if read_data is not None:
            self.read_data = read_data
            
        self.read_data(input)       # Use provided read function
        self.split_data()                  # Split into train/test as needed
        self.print_set_sizes()

            
    def split_data(self):
        """
        Split data into testing and training sets based on options

        OPTIONS:
            train : Use 100% of the datalines for training

            test  : Use 100% of the datalines for testing

            trainAndTest : Use a portion of the lines for training (0.1 by default)
                and the rest for tesing.  Training/Testing lines will be selected at random
        """
        if self.get('train'):                # 100% for Training
            #                                # For more on 'self.get(option_name)' see objects.py Object class
            self.train = self.data

        elif self.get('test'):               # 100% for Testing
            self.test = self.data

        elif self.get('trainAndTest'):       # Training AND Testing
            if self.get('validation'):
                self.train, self.test                   = train_test_split(self.data, test_size=0.09)
            else:
                self.train, self.test, self.validation  = train_test_split(self.data, test_size=0.08, validation_size=0.01)
            self.print_set_sizes()

        else:
            sys.stderr.write("No task selected. Exiting...\n")
            exit()

            
    def print_set_sizes(self):
        """
        Print info about the training and test sets
        """
        n_train = self.train.shape[0]         # Lines of input text for training
        n_test  = self.test.shape[0]          # Lines of input text for testing

        sys.stderr.write('Training set size:  %d\n'% n_train)
        sys.stderr.write('Test set size:      %d\n'% n_test)


    def get_training_XY(self, signal=None):
        """
        For use in modules where X and Y are handled separately

        Returns
        -------
        X : DataFrame

        Y : Series

        """
        if signal is None:
            signal = self.get('signal')
        
        X = self.train.loc[:, self.train.columns != signal]
        Y = self.train[signal]

        X = X.as_matrix()
        Y = Y.tolist()
        
        return X, Y

            
    def get_testing_XY(self, signal=None):
        """
        For use in modules where X and Y are handled separately

        Returns
        -------
        X : DataFrame

        Y : Series

        """
        if signal is None:
            signal = self.get('signal')
            
        X = self.test.loc[:, self.test.columns != signal]
        Y = self.test[signal]
        
        X = X.as_matrix()
        Y = Y.tolist()
        
        return X, Y


    def balance_with_synthetic_minority_oversampling(self, X, Y):
        """
        For the minority set, iterate over datapoints:
            - select 1
            - find k nearest neighbors
            - choose j of them
            - generate a new point halfway between the current point and each of j

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        verbose = False
        X2 = Y2 = None

        counts = self.get_class_counts(Y)
        print("Class counts:", counts)
        
        # For now, only works with binary classes (1, 0)
        if counts[0] > counts[1]:

            # synthetically increase class 1
            X2, Y2 = self.synthetic_samples(X, Y, 1, counts[0] - counts[1])
            
        elif counts[0] < counts[1]:
            # synthetically increase class 0
            X2, Y2 = self.synthetic_samples(X, Y, 0, counts[1] - counts[0])

        else:
            return X, Y

        X2.columns = X.columns   # Must be done first to allow the concat process
        X = pd.concat([X, X2])
        Y = Y.append(Y2)
        
        counts = self.get_class_counts(Y)
        print("Class counts after synthetic minority oversampling:", counts)

        return X, Y


    def balance_with_majority_undersampling(self, X, Y):
        """
        For the majority set select a subset such that both classes have the same number of examples

        Note: only works for binary classification at this time.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        verbose = False
        X2 = Y2 = None

        counts = self.get_class_counts(Y)
        print("Class counts:", counts)
        
        # For now, only works with binary classes (1, 0)
        if counts[0] > counts[1]:
            X, Y = self.undersample(X, Y, 0, counts[1])  # decrease class 0
            
        elif counts[0] < counts[1]:
            X, Y = self.undersample(X, Y, 1, counts[0])  # decrease class 1

        else:
            return X, Y

        counts = self.get_class_counts(Y)
        print("Class counts after majority undersampling:", counts)

        return X, Y


    def nearest_neighbors(self, dataframe, i, n=5):
        """
        Within the dataframe df, find n other rows nearest to x

        It returns the closest match, dropping the first because it is assumed to be identical

        """
        df = deepcopy(dataframe)
        n += 1  # because we're dropping the closest match
        arr = df.values
        def fsi_numpy(item_id):
            tmp_arr = arr - arr[item_id]
            tmp_ser = np.sum( np.square( tmp_arr ), axis=1 )
            return tmp_ser

        df['dist'] = fsi_numpy(i)
        df = df.sort_values('dist').head(n).drop(labels=['dist'], axis=1)

        return df.iloc[1:] 


    def synthetic_samples(self, X, Y, C, N):
        """
        Generate N synthetic samples of class C

        Parameters
        ----------
        X : pandas.DataFrame
            input samples

        Y : pandas.Series
            supervised output

        C : int
            specifies a class

        N : int
            number of new samples to generate
        """
        verbose = False
        X2 = []  # the generated samples
        Y2 = []

        # Generate dataframe of only samples where Y=C
        arr_C = []
        for i,row in X.iterrows():
            if int(Y.iloc[i]) == C:
                arr_C.append(row.values)

        X_C = pd.DataFrame(arr_C)    # This dataframe only has samples where Y is of class C
        
        # Generate some sample indices and create synthetic examples close to these
        indices = range(X_C.shape[0])
        arr_S = []              # To hold the new synthetic datapoints
        Y_S   = pd.Series([])   # These will all equal C
        C = pd.Series([C])      # To make it the right format
        while len(arr_S) < N:

            i = random.choice(indices)
            x = X_C.iloc[i]
            neighbors = self.nearest_neighbors(X_C, i, 2)
            chosen_one = neighbors.sample().iloc[0]

            arr_S.append( self.average_two_rows(x, chosen_one) )
            Y_S = Y_S.append(C)
            
            if verbose:
                print ('neighbors:', neighbors)
                err(['x:', x, type(x), x.size])
                err(['chosen one:', chosen_one, type(chosen_one), chosen_one.size])
                err([type(Y_S), Y_S.size])
            
        X_S = pd.DataFrame(arr_S)
        return X_S, Y_S
        

    def undersample(self, X, Y, C, N):
        """
        Limit samples of class C to no more than N

        example call:
        X2, Y2 = self.undersample(X, Y, 1, diff)  # decrease class 1

        Parameters
        ----------
        X : pandas.DataFrame
            input samples

        Y : pandas.Series
            supervised output

        C : int
            specifies a class

        N : int
            number of samples to keep
        """
        verbose = False

        # Find index of each example of C, but also keep track of the others (we will keep all of them)
        indices_to_keep = Set([])
        indices_C = []
        for i,row in X.iterrows():
            y = int(Y.iloc[i])
            if y == C:
                indices_C.append(i)
            else:
                indices_to_keep.add(i)

        # Add a random sample of the C indices to the keep set
        to_keep_C = Set(random.sample(indices_C, N))
        indices_to_keep = indices_to_keep.union(to_keep_C)

        # Iterate over original data, pulling out what to keep
        arr_X = []
        arr_Y = []
        for i,row in X.iterrows():
            y = int(Y.iloc[i])
            if i in indices_to_keep:
                arr_X.append(row.values)
                arr_Y.append(y)

        X2 = pd.DataFrame(arr_X)    # This dataframe only has the selected rows
        Y2    = pd.Series(arr_Y)
        
        return X2, Y2
        

    def average_two_rows(self, a, b):
        """
        Takes two Series of the same length and performs element-wise averaging.  Used for SMOTE

        Parameters
        ----------
        a, b : pandas.Series

        Returns
        -------
        numpy array

        """
        verbose = False
        a = a.values
        b = b.values
        if verbose: err([a, type(a), len(a), b, type(b), len(b)])
        assert(len(a) == len(b))
        c = np.mean([a, b], axis=0)
        if verbose:
            np.set_printoptions(precision=2)
            err([a, b, c])
        
        return c

    
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
    
