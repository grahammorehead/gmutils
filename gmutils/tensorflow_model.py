""" tensorflow_model.py

    Tools for building ML models with TensorFlow

"""
import os, sys, re
import numpy as np
import tensorflow as tf

from gmutils.utils import err, argparser
from gmutils.model import Model

################################################################################
# CONFIG

default = {
    'batch_size'         : 100,
    'epochs'             : 1,
    'learning_rate'      : 0.01,
}

################################################################################
# OBJECTS

class TensorflowModel(Model):
    """
    An object to assist in the training, storage, and utilizating of TensorFlow models.

    Attributes  (depends on subclass)
    ----------
    graph : the initial default tf.Graph

    model : a TensorflowGraph object as defined in tensorflow_graph.py
        Not to be confused with a tf.Graph.  This is a connected graph of Tensors

    train_dir : str
        Path to where training files are stored

    eval_dir : str
        Path to where eval files are stored

    model_dir : str
        Path to where the model is stored

    model_file : str
        File path where the model is stored

    global_initializer : TF variable initializer

    iterator : iterable providing data

    optimizer = tf.train optimizer

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)
        self.graph = tf.get_default_graph()
        

    def initialize(self, sess, options={}):
        """
        Initialize all or needed variables for a given session
        """
        if options.get('only_uninitialized'):
            global_vars          = tf.global_variables()
            is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

            # print [str(i.name) for i in not_initialized_vars] # only for testing
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

        else:
            init = tf.global_variables_initializer()
            sess.run(init)


    def assert_graph(self):
        """
        Asserts that the current default graph is what it should be
        """
        assert tf.get_default_graph() is self.graph


    def fit(self, iterator):
        """
        Iterate through data training the model
        """
        for _ in range(self.get('epochs')):
            while True:
                try:
                
                    sess.run(self.optimizer, feed_dict=self.model.feed_dict)
                    
                    # Get the next batch of data for training
                    datum = next(iterator)
                    
                except StopIteration:
                    break

                
################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
