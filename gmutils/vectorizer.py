""" vectorizer.py

    Code and objects to vectorize text

"""
import os, sys, re
from copy import deepcopy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, chi2

from gmutils.utils import err, argparser
from gmutils.model import pandasize
from gmutils.objects import Object
from gmutils.normalize import clean_spaces
from gmutils.document import Document

################################################################################
##  DEFAULTS   ###

default_underlying = ['lemmas', 'pos']

default_max_feat = {                     # Length of the section of the output vector associated with each underlying vectorizer
    'text'      : 20000,
    'lemmas'    :  8000,
    'pos'       :  4000,
    'entities'  :  1000,
    'entity_types'  : 500
    }
    
default_min_df = {                       # Minimum document frequency for each underlying vectorizer
    'text'      : .0002,
    'lemmas'    : .0001,
    'pos'       : .001,
    'entities'  : .00001,
    'entity_types'  : .001
    }

default_max_df = {                       # Maximum document frequency for each underlying vectorizer
    'text'      : 0.98,
    'lemmas'    : 1.0,
    'pos'       : 1.0,
    'entities'  : 1.0,
    'entity_types'  : 1.0
    }

default_stop_words = None

float_features = []

int_features   = ['entity_count', 'word_count']

str_features   = []
# Note: These are converted to numbers in the function string_class_to_number()

    
################################################################################
##   OBJECTS   ###


class Vectorizer(Object):
    """
    An object to manage converting lines of text into vectors.  This object manages the underlying vectorizer(s) and performs dimensionality reduction.

    Objective: convert lines of text of varying length into a vectors of equal length.

    Intuitive notion: lines of text which are most similar should generate vectors with the highest cosine similarity.

    This object can be serialized to disk
    """
    
    def __init__(self, options={}):
        """
        Initialize the object

        Attributes
        ----------
        underlying : array of str
            Names of the underlying vectorizers (and their associated feature selectors).
            could be chosen from:
                1. text
                2. lemmas
                3. pos
            NOTE: There may be a way to make this more extensible but for now these three are hard-coded
        
        vectorizers : dict of underlying sklearn.feature_extraction.text vectorizers
            Each vectorizer is used to generate a part of the output vector.  In this way, different parts of the output vector can propagate different
            types of signals, e.g. The first 100 elements might be the output of TFIDF on the POS tags, while the next 1000 are the ouput of TFIDF on lemmas.
            The underlying vectorizers include:
                1. TFIDF on the input text
                2. TFIDF on the lemmatized text
                3. TFIDF on the POS tags

        selectors : dict of sklearn.feature_selection selectors
            These are used for dimensionality reduction.
            As with the vectorizers, there is one for each of 'text', 'lemmas', and 'pos'
        """

        # initializations
        self.set_options(options)       # For more on 'self.set_options()' see objects.py DSObject class
        self.selectors = {}
        self.vectorizers = {}

        ##  VECTORIZER PARAMETERS
        if self.get('underlying'):
            self.underlying = self.get('underlying')
        else:
            self.underlying = default_underlying

        if not self.get('max_feat'):
            self.set('max_feat', default_max_feat)

        if not self.get('min_df'):
            self.set('min_df', default_min_df)

        if not self.get('max_df'):
            self.set('max_df', default_max_df)

        if not self.get('stop_words'):
            self.set('stop_words', default_stop_words)


    def fit_transform(self, input, Y=None, verbose=False):
        """
        Configures the Vectorizer.

        Contruct a TFIDF vectorizer to convert a string into a vector.  Incorporates old lexical method.

        Paramters
        ---------
        input : array of str

        Y : array of values
            Array, where each element is a supervised classification
            NOTE: setting Y to "ignore" will allow this function to instantiate and fit a vectorizer, but will inhibit the feature selector

        Returns : matrix
            vectorized version of input
        """
        # Use locals so we don't lose original information
        underlying = deepcopy(self.underlying)

        min_df   = self.get('min_df')
        max_df   = self.get('max_df')
        max_feat = self.get('max_feat')
        
        parsed   = get_parsed_input_for_underlying(input, underlying)

        # If Requested, generate a set of "enhanced" features, computed from the existing ones
        if self.get('enhanced_features'):
            enhance_features = self.get('enhance_features')   # retrieve this function from the options
            parsed, underlying = enhance_features(parsed, underlying)   # Add some features

        if self.get('features_to_drop'):
            features_to_drop = self.get('features_to_drop')
            if len(features_to_drop) > 0:
                parsed, underlying = drop_features(parsed, underlying, features_to_drop)   # Add some features
        
        # Costruct and use Vectorizers and Feature Selectors
        X = None        # The output vector
        for u in underlying:
            X1 = None   # This part of the output vector

            # NOTE: for this section to properly vectorize incoming data, the feature name should either be a string having words, or should be
            #       included in one of these feature lists:
            #           - float_features
            #           - int_features
            #           - str_features
            
            # If <u> corresponds to a floating-point value, just concat it as its own part of the vector
            if u in float_features:    # (See: defaults section at top of tile)
                X1 = parsed[u]
                print('Vectorizing', u, '...')
                
            # If <u> corresponds to an integer value, just concat it as its own part of the vector
            elif u in int_features:    # (See: defaults section at top of tile)
                X1 = parsed[u]
                print('Vectorizing', u, '...')
                
            # If <u> corresponds to a text value, convert it to a number and then concat it to the vector
            elif u in str_features:    # (See: defaults section at top of tile)
                X1 = string_class_to_number(u, parsed[u])
                print('Vectorizing', u, '...')
                
            # Pre-fit, Just Transform
            elif Y is None:
                if verbose:
                    sys.stderr.write("Using %s Vectorizer ...\n"% u)
                p = parsed[u]
                try:
                    X1 = self.vectorizers[u].transform(p)   # Transform into a vector
                    if not self.get('hashing_vectorizer'):
                        try:
                            X1 = self.selectors[u].transform(X1)
                        except:
                            raise
                            pass
                except:
                    raise
                    return None

            # Fitting and Transforming str-based features having "words" of one kind or another.
            else:
                if verbose:
                    sys.stderr.write("Instantiating %s Vectorizer ...\n"% u)

                if self.get('hashing_vectorizer'):
                    self.vectorizers[u] = HashingVectorizer(n_features=max_feat[u], lowercase=True, analyzer=u'word', strip_accents='unicode', binary=False, stop_words=self.get('stop_words'), ngram_range=(1,4), norm=u'l2')
                    #                     HashingVectorizer( OTHER OPTIONS: alternate_sign=True, non_negative=False )
                else:
                    self.vectorizers[u] = TfidfVectorizer(analyzer=u'word', sublinear_tf=True, strip_accents='unicode', max_df=max_df[u], min_df=min_df[u], max_features=max_feat[u], vocabulary=None, binary=False, stop_words=self.get('stop_words'), ngram_range=(1,4), norm=u'l2', use_idf=True, smooth_idf=True)

                if verbose:
                    sys.stderr.write("Fitting %s Vectorizer ...\n"% u)
                X1 = self.vectorizers[u].fit_transform( parsed[u] )      # Compute vocabulary, idf, term-document matrix, then transform into a vector
                if verbose:
                    sys.stderr.write("\tDone.\n")

                # IF using a selector for dimensionality reduction
                if len(Y) == len(parsed[u])  and  not self.get('hashing_vectorizer'):
                    n0 = X1.shape[1]                                                              # Length of incoming vector
                    max_feat[u] = min(int(0.8 * n0), max_feat[u])                                 # Max output length
                    sys.stderr.write("\tReducing featureset %d -> %d ...\n"% (n0, max_feat[u]))
                    # self.selectors[u] = SelectKBest(mutual_info_classif, k=max_feat[u])           # Dimensionality reduction by selection
                    self.selectors[u] = SelectKBest(chi2, k=max_feat[u])           # Dimensionality reduction by selection
                    X1 = self.selectors[u].fit_transform(X1, Y)
                    sys.stderr.write("\tDone.\n")
                    
                else:
                    n0 = X1.shape[1]                                                              # Length of incoming vector
                    sys.stderr.write("\tFeatureset has %d features.\n"% (n0))

            if len(X1.shape) == 1:
                X1 = X1.values.reshape((X1.shape[0],1))

            # pandasize ?
            
            # horizontally concatenate output vectors
            if X1 is None:
                err(['X1 is None'], {'exit':True})
            if X is None:
                X = X1
            else:
                X = pd.concat([X, X1], axis=1)
                X.fillna(0.0, inplace=True)

        return X


    def get_feature_names(self):
        feature_names = []
        for u in self.underlying:
            feature_names.extend(self.vectorizers[u].get_feature_names())
        return feature_names

            
    def transform(self, input, verbose=False):
        """
        Use vectorizer to convert a string into a vector

        Paramters
        ---------
        input : array of str

        Returns : matrix
            vectorized version of 'input'
        """
        X1 = self.fit_transform(input)
        if verbose:
            n0 = X1.shape[1]                                                              # Length of incoming vector
            sys.stderr.write("\tFeatureset has %d features.\n"% (n0))
        return X1


################################################################################
# FUNCTIONS


def get_parsed_input_for_underlying(input, underlying):
    """
    Takes various kinds of input and generates a dict to be used by the vectorizer as parsed input.

    Parameters
    ----------
    input : Dataframe or 
        Treat each input type differently
    
    underlying : array of str
        The names of the inputs which will be used
    """
    verbose = False
    parsed = {}

    for u in underlying:

        if isinstance(input, list):
            parsed[u] = input
        else:
            parsed[u] = input[u]

    return parsed


def drop_features(parsed, underlying, features_to_drop):
    """
    Create a smaller set of features to be analyzed.

    Example call:
        parsed, self.underlying = drop_features(parsed, self.underlying)

    Parameters
    ----------
    parsed : dict of various

    underlying : array of str

    features_to_drop : array of str
        List of features to drop, by name
    
    Returns
    -------
    parsed, underlying

    """

    for feature in features_to_drop:
        try:
            parsed.pop(feature, None)
            underlying.remove(feature)
        except:
            pass
    
    return parsed, underlying



################################################################################
# MAIN

if __name__ == '__main__':
    try:
        parser = argparser({'desc': "Tools to vectorize text: vectorizer.py"})
        #  --  Tool-specific command-line args may be added here
        args = parser.parse_args()   # Get inputs and options

        print(__doc__)

    except Exception as e:
        print(__doc__)
        err([], {'exception':e, 'exit':True})

        
################################################################################
################################################################################
    
