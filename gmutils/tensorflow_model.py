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
    'dtype'              : tf.float16,
    # 'dtype'              : tf.float32,
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
        self._monitor = None
        

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


    def run(self, sess, targets):
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
                
            if self._monitor:
                epoch    = self._monitor.get('epoch')
                step     = self._monitor.get('step')
                loss_val = output[-1]

                # Output training state to the command line
                if not self.get('silent'):
                    last_update_line = self._monitor.get('update_line')
                    update_line =  "%s (e %d, b %d, s %d) [loss %0.16f] {lr %08f}"% (
                        self._monitor['progress'],
                        epoch,
                        self._monitor['i'],
                        step,
                        loss_val,
                        self._monitor.get('learning_rate') )
                    
                    if last_update_line is not None:
                        sys.stderr.write('\b' * (len(last_update_line) + 1))   # 0 for no newline
                    sys.stderr.write('\n')
                    sys.stderr.write(update_line)
                    sys.stderr.flush()
                    self._monitor['update_line'] = update_line
                    
                return output
            
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

    
    def get_good_model(self):
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
            models = self.model_files_by_loss()   # sorted from lowest loss to highest
            x = beta_choose(len(models))
            loss_val, filepath = models[x]
            model = json_load_gz(filepath)
            return model
        except:
            return None
    
    
    def get_recent_model(self):
        """
        Randomly select and retrieve a recent model via a Beta distribution.  This will usually select the most recent model but not always.
        May help to escape from local minima.  Although the currently selected loss function is convex for any given batch, since the graph
        is redrawn for each batch, I cannot yet confirm global convexity

        A separate process will attempt to winnow out the models with higher loss.

        Returns
        -------
        dict { var_name : numpy array }
            Use these values to assign arrays to tensors

        """
        try:
            models = self.model_files_by_timestamp()   # sorted, most recent first
            x = beta_choose(len(models))
            loss_val, filepath = models[x]
            model = json_load_gz(filepath)
            return model
        except:
            return None
    
    
    def load_best_model(self, sess):
        """
        If present, restore best existing model

        Returns
        -------
        boolean
            True if the model loaded without error
        """
        model = self.get_best_model()
        if model is not None:
            return self.load(sess, model)
        
        
    def load_good_model(self, sess):
        """
        If present, load a "good" model (selected via beta_choose()), and restore it to the live session.  Randomly chooses a low-loss model.
        To be used either at the beginning of a training session or mid-session if desired.

        Returns
        -------
        boolean
            True if the model loaded without error
        """
        model = self.get_good_model()
        if model is not None:
            return self.load(sess, model)
        
        
    def load_recent_model(self, sess):
        """
        If present, load a recent model (selected via beta_choose()), and restore it to the live session.  Randomly chooses a recent model.
        To be used either at the beginning of a training session or mid-session if desired.

        Returns
        -------
        boolean
            True if the model loaded without error
        """
        model = self.get_recent_model()
        if model is not None:
            return self.load(sess, model)
        
        
    def load_most_recent_model(self, sess):
        """
        If present, load a recent model (selected via beta_choose()), and restore it to the live session.  Randomly chooses a recent model.
        To be used either at the beginning of a training session or mid-session if desired.

        Returns
        -------
        boolean
            True if the model loaded without error
        """
        model = self.get_most_recent_model()
        if model is not None:
            return self.load(sess, model)
        
        
    def model_files_by_loss(self):
        """
        List all available model files by loss
        """
        models = {}
        model_files = read_dir(self.get('model_dir'))
        for f in model_files:
            if re.search('^loss_', f):
                lv = re.sub(r'^loss_', '', f)
                lv = re.sub(r'\.json\.gz$', '', lv)
                loss_val = float(lv)
                models[loss_val] = self.get('model_dir') +'/'+ f
                
        return sorted(models.items(), key=lambda x: x[0])


    def model_files_by_timestamp(self):
        """
        List all available model files by timestamp, most recent first
        """
        models = {}
        model_files = read_dir(self.get('model_dir'))
        for f in model_files:
            if re.search('^loss_', f):
                filepath = self.get('model_dir') +'/'+ f
                ts = os.path.getmtime(filepath)
                models[ts] = filepath
                
        return sorted(models.items(), key=lambda x: x[0], reverse=True)  # "reverse", because we want the highest timestamps (most recent) first


    def get_best_model(self):
        """
        Get the best model (based on its loss)

        Returns
        -------
        dict { var_name : numpy array }
            Use these values to assign arrays to tensors

        """
        try:
            loss_val, filepath = self.model_files_by_loss()[0]   # Just take first pair
            model = json_load_gz(filepath)
            # print("\nLoading BEST model:", filepath, "(%s)"% file_timestamp(filepath))
            return model
        
        except:
            return None
    
    
    def get_most_recent_model(self):
        """
        Get the most recent model (based on timestamp)

        Returns
        -------
        dict { var_name : numpy array }
            Use these values to assign arrays to tensors

        """
        verbose = False
        try:
            models = self.model_files_by_timestamp()   # sorted, most recent first
            if verbose:
                for i, model in enumerate(models):
                    print(i, ":", model[1])
            loss_val, filepath = models[1]             # Just take second one in case most recent is still downloading
            model = json_load_gz(filepath)
            # print("\nLoading RECENT model:", filepath, "(%s)"% file_timestamp(filepath))
            return model
        
        except:
            return None

    
    def clear_chaff(self):
        """
        Get a list of all models sorted by loss.  Keep MAX, and delete the rest
        """
        verbose = False
        MAX = 500
        
        # models = self.model_files_by_loss()        # sorted, lowest-loss first
        models = self.model_files_by_timestamp()   # sorted, most recent first
        if len(models) > MAX:
            try:
                to_delete = models[MAX:]
                for model in to_delete:
                    loss_val, filepath = model
                    if verbose:
                        t = file_timestamp(filepath)
                        print("rm (%s):"% t, filepath)
                    os.remove(filepath)
            except:
                pass
    

    def sum_sqdiff(self, labels, preds):
        """
        Use tensors to find the average squared difference between values coming from two arrays of tensors

        Boost the error of the answer to put focus on it.

        NOTE: Be careful to keep this funciton in sync with the following one.
        """
        loss = None
        for i, x in enumerate(labels):
            y = preds[i]

            sqdiff = tf.squared_difference(x, y)
            
            if loss is None:
                loss = sqdiff
            else:
                loss = tf.add(loss, sqdiff)   # Just add up all the square differences
                
        return loss

    
    def sum_sqdiff_with_values(self, labels, preds):
        """
        Use tensors to find the average squared difference between values coming from two arrays of tensors

        Boost the error of the answer to put focus on it.

        NOTE: Be careful to keep this funciton in sync with the following one.
        """
        loss = None
        for i, x in enumerate(labels):
            y = preds[i]

            sqdiff = (x - y)**2
            
            if loss is None:
                loss = sqdiff
            else:
                loss = loss + sqdiff   # Just add up all the square differences
                
        return loss

    
    def boosted_sqdiff(self, labels, preds):
        """
        Use tensors to find the average squared difference between values coming from two arrays of tensors

        Boost the error of the answer to put focus on it.

        NOTE: Be careful to keep this funciton in sync with the following one.
        """
        max_label_i = tf.argmax(labels, axis=0)
        D = []
        for i, x in enumerate(labels):
            y = preds[i]

            sqdiff = tf.squared_difference(x, y)
            D.append(sqdiff)

        ansT = tf.gather(D, max_label_i)  # loss attributed to the answer
        sumT = tf.add_n(D)                # all loss
        nT   = tf.constant(len(D), dtype=self.get('dtype'))
        divT = tf.divide(sumT, nT)
        
        return tf.add(divT, ansT)

    
    def boosted_sqdiff_with_values(self, labels, preds):
        """
        Recompute the loss fuction mirrored above.  Be careful to keep these two functions in sync.
        """
        max_label_i = np.argmax(labels, axis=0)[0][0]
        D = []
        for i, x in enumerate(labels):
            y = preds[i]
            label = x[0][0]
            pred  = y[0][0]
            sqdiff = (x - y)**2
            D.append(sqdiff)

        ansT = D[max_label_i]  # loss attributed to the answer
        sumT = np.sum(D)       # all loss
        nT   = len(D)
        divT = sumT / nT
        
        return divT + ansT

    
    
################################################################################
# FUNCTIONS

def beta_choose(N):
    """
    Use an almost-Beta function to select an integer on [0, N]

    """
    x = int( np.random.beta(1, 128) * N+1 )
    if x > 0:
        x -= 1
    return x


################################################################################
##   MAIN   ##

if __name__ == '__main__':
    parser = argparser_ml({'desc': "Tools to train and use TensorFlow models: tensorflow_model.py"})
    args = parser.parse_args()   # Get inputs and options

        
################################################################################
################################################################################
