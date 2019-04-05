""" objects.py

    Using these Objects as superclasses will help produce cleaner, simpler code.

"""
import sys, os, re
import argparse
import types

from .utils import mkdir, err, isTrue

################################################################################
# CONFIG

default = {}

################################################################################
# OBJECTS

class Object(object):
    """
    A custom subclass of object to assist in the handling of options within code and
    to result in cleaner code overall

    """
    def __init__(self, options=None):
        """
        Instantiate the object and set options
        """
        self.set_options(options, default)

        
    def set_options(self, options=None, default=default):
        """
        The purpose of this and related functions is to enable getting and setting of options without having to check first if a dict key exists-- results in cleaner code.

        Parameters
        ----------
        options : dict
            In this case each key in this dict will be converted to an attribute. The name of each key must be only lower case with underscores

            or  : None
            In this case an empty object is created

            or  : argparse.Namespace
            In this case it iterates over non internal attributes accruing them to this object
        """
        verbose = False
        if verbose:  sys.stderr.write("default = %s\n"% str(default))
            
        try:   # Confirm that self._value_ exists
            if not hasattr(self, '_value_'):
                self._value_ = {}
        except AttributeError:
            self._value_ = {}
        except:
            self._value_ = {}

        self.options = options
        
        if options is not None:

            if isinstance(options, dict):
                for key,val in options.items():
                    if not key == 'input':
                        if verbose:  sys.stderr.write("key, val = { %s : %s }\n"% (str(key), str(val)))
                        self.set(key, val)

            elif isinstance(options, Options):
                for key,val in options._value_.items():

                    # Skip some keys that shouldn't be absorbed into the next object
                    if re.search('^_', key):
                        continue
                    if key == 'input'  or  key == 'options':
                        continue
                    if isinstance(val, types.MethodType):
                        continue
                    
                    self.set(key, val)

            elif isinstance(options, argparse.Namespace):
                for key in dir(options):

                    # Skip some keys that shouldn't be absorbed into the next object
                    if re.search('^_', key):
                        continue
                    if key == 'input'  or  key == 'options':
                        continue
                    val = getattr(options, key)
                    if isinstance(val, types.MethodType):
                        continue
                    
                    self.set(key, val)

            else:
                print('(gmutils/objects.py) ERROR: options object is of type:', type(options))
                exit()

        self.set_missing_attributes(default)

        self.mkdirs()  # Some settings require that a given directory exists
                    

    def set_missing_attributes(self, attributes=None):
        """
        Set some missing options using a dict of defaults.
        Some options may have been missing because they either weren't serializable or simply weren't specified.

        """
        verbose = False
        if attributes is not None:
            for param, val in attributes.items():
                if not self.get(param):
                    if verbose:
                        if param not in ['vocab']:
                            sys.stderr.write(" param, val = { %s : %s }\n"% (str(param), str(val) ))
                    self.set(param, val)
                    
        
    def override_attributes(self, attributes=None):
        """
        Set some missing options using a dict of defaults.  Override any existing values.
        
        """
        if attributes is not None:
            for param in attributes.keys():
                self.set(param, attributes[param])

        
    def set(self, key, val):
        """
        Set key/val pair (for options)

        Parameters
        ----------
        key : str

        val : anything
        """
        self._value_[key] = val

        
    def get(self, key):
        try:
            return self._value_[key]
        except KeyError:
            # raise AttributeError(key)
            return None
        except:
            self._value_ = {}
            return None


    def __repr__(self):
        out = {}
        for k,v in self._value_.items():
            out[k] = str(v)
        return out

        
    def __str__(self):
        return str( self.__repr__() )


    def isTrue(self, key):
        """
        To enable shorter code for determining if an option is set
        """
        try:
            out = self.get(key)
            if out:
                return out
            return False
        except:
            pass

        return False


    def isVerbose(self):
        """
        To enable shorter code for determining if an option is set
        """
        return self.isTrue(options, 'verbose')


    def mkdirs(self):
        """
        For some options, a directory is called for.  Ensure that each of these exist.

        """
        check = ['default_dir', 'output_dir', 'model_dir']
        for c in check:
            if self.get(c):
                mkdir(self.get(c))
        
    
    def get_config(self, options=None):
        """
        Generate a dict of the option settings.

        Returns
        -------
        dict

        """
        config = {}
        for param, val in self._value_.items():

            if isTrue(options, 'serializable'):
                if is_jsonable(val):
                    config[param] = val
            else:
                config[param] = val
            
        return config

    
    def done(self):
        """
        Used to make sure that a given function is only called once or a limited number of times.

        Returns
        -------
        int : the number of times 'caller' has been called for this object

        """
        caller = sys._getframe(1).f_code.co_name   # name of the function in which 'done()' was called
        tracker = '_DONE_' + caller                # name of stored key to track how many times 'caller' has been called for this object
        so_far = self.get(tracker)                 # number of times 'caller' has been called so far
        try:
            self.set(tracker, so_far + 1)          # increment the number of times
        except:
            self.set(tracker, 1)                   # first time

        return so_far
        

    def clear_done(self):
        """
        Used to enable a given function to be called again

        Returns
        -------
        int : the number of times 'caller' has been called for this object

        """
        caller = sys._getframe(1).f_code.co_name   # name of the function in which 'done()' was called
        tracker = '_DONE_' + caller                # name of stored key to track how many times 'caller' has been called for this object
        self.set(tracker, 0)


    def get_type(self):
        """
        Returns a string with just the name of the instantiated class
        """
        T = str(type(self))
        T = re.sub(r"^<class '", "", T)
        T = re.sub(r"'>$", "", T)
        _, T = T.split('.')
        
        return T
        

class Options(Object):
    """
    A custom subclass of object>Object to assist in the handling of options outside of object code

    """
    def __init__(self, options):
        self.set_options(options)


        
################################################################################
################################################################################
