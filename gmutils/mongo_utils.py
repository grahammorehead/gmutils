""" mongo_utils.py

Helper functions for Mongo

"""
import os, sys, re
from pprint import pprint
from pymongo import MongoClient

from gmutils.utils import argparser

################################################################################

def test(host='localhost', port=27017, db_name='default', collection_name='default'):
    client = MongoClient(host, port)
    db = client[db_name]
    print("Loaded db", db_name,"  type:", type(db))
    print("Collections:", db.collection_names())
    coll = db[collection_name]
    print("Loaded collection", collection_name,"  type:", type(coll))


def mongo_iterator(db_name='default', collection_name='default', host='localhost', port=27017):
    """
    Iterate over all docs in a Mongo DB

    Parameters
    ----------
    host : str

    port : int

    db_name : str

    collection_name : str

    Returns
    -------
    iterator

    """
    verbose = True
    if verbose:
        print("Mongo host:", host, "    port:", port,  "   type:", type(port))
        print("Will iterate on collection:", collection_name)
    
    client = MongoClient(host, port)
    db = client[db_name]
    coll = db[collection_name]
    return coll.find()


def mongo_find_one(db_name='default', collection_name='default', host='localhost', port=27017):
    """
    Iterate over all docs in a Mongo DB

    Parameters
    ----------
    host : str

    port : int

    db_name : str

    collection_name : str

    Returns
    -------
    iterator

    """
    verbose = True
    if verbose:
        print("Mongo host:", host, "    port:", port,  "   type:", type(port))
        print("Will iterate on collection:", collection_name)
    
    client = MongoClient(host, port)
    db = client[db_name]
    coll = db[collection_name]
    return coll.find_one()


################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Helper functions for Mongo: mongo_utils.py"})

    #  --  Tool-specific command-line args may be added here
    parser.add_argument('--list', help='List all databases', required=False, action='store_true')
    args = parser.parse_args()   # Get inputs and options
    
    if args.list:
        test(db_name='db', collection_name='collection')   # just dummy names

        
################################################################################
################################################################################