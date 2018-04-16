""" tensorflow_model.py

    Tools for building ML models with TensorFlow

"""
import os, sys, re
import time
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
    # 'dtype'              : tf.float16,
    'dtype'              : tf.float32,
}

################################################################################
# OBJECTS

class TensorflowModel(Model):
    """
    An object to assist in the training, storage, and utilizating of TensorFlow models.

    Attributes  (depends on subclass)
    ----------
    graph : the tf.Graph where all Variables are contained/connected

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


    def run(self, sess, targets, _monitor=None):
        """
        Attempt to run in the current session.  When fails, wait one second and try again.

        Necessary because grabbing the GPU when another TF process is on it can be disasterous

        Parameters
        ----------
        sess : tf.Session

        targets : Tensors

        """
        try:
            output = sess.run(targets, feed_dict=self.model.feed_dict)
                
            if _monitor:
                epoch    = _monitor.get('epoch')
                step     = _monitor.get('step')
                loss_val = output[-1]

                # Output training state to the command line
                if not self.get('silent'):
                    last_update_line = _monitor.get('update_line')
                    update_line =  "%s (e %d, b %d, s %d) [loss %0.16f] {lr %08f}"% (_monitor['progress'], epoch, _monitor['i'], step, loss_val, _monitor.get('learning_rate'))
                    if last_update_line is not None:
                        sys.stdout.write('\b' * len(last_update_line))
                    else:
                        sys.stdout.write('\n')
                    sys.stdout.write(update_line)
                    sys.stdout.flush()
                    _monitor['update_line'] = update_line
                    
                return output, _monitor
            
            else:    # Without monitoring
                return output
            
        except:
            raise
                
        
    def fit(self, iterator):
        """
        Iterate through data training the model
        """
        for _ in range(self.get('epochs')):
            while True:
                try:
                    sess.run(self.optimizer, feed_dict=self.model.feed_dict)
                    datum = next(iterator)   # Get the next batch of data for training
                except StopIteration:
                    break


    def learning_rate_by_epoch(self, epoch):
        """
        To compute a new learning rate for each epoch (lower each time, of course)

        Parameters
        ----------
        epoch : int

        Returns
        -------
        float

        """
        return self.get('learning_rate') * (0.8 ** (epoch-1))


    def avg_sqdiff(self, X, Y):
        """
        Use tensors to find the average squared difference between values coming from two arrays of tensors
        """
        D = []
        for i, x in enumerate(X):
            y = Y[i]

            sqdiff = tf.squared_difference(x, y)
            D.append(sqdiff)

        sumT = tf.add_n(D)
        nT   = tf.constant(len(D), dtype=self.get('dtype'))
        divT = tf.divide(sumT, nT)
        
        return divT


    def max_sqdiff(self, X, Y):
        """
        Use tensors to find the max squared difference between values coming from two arrays of tensors
        """
        D = []
        for i, x in enumerate(X):
            y = Y[i]

            delta = tf.abs( tf.squared_difference(x, y) )
            D.append(delta)

        max_i = tf.argmax(D, axis=0)
        maxT  = D[i]
            
        return maxT
                
    
################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
