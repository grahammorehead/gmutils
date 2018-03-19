""" mongo_utils.py

Helper functions for Mongo

"""
import os, sys, re
from pprint import pprint
from pymongo import MongoClient
from urllib.parse import quote_plus

from gmutils.utils import argparser, err

################################################################################

def test(host='localhost', port=27017, db_name='default', collection_name='default'):
    client = MongoClient(host, port)
    db = client[db_name]
    print("Loaded db", db_name,"  type:", type(db))
    print("Collections:", db.collection_names())
    coll = db[collection_name]
    print("Loaded collection", collection_name,"  type:", type(coll))


def get_mongo_client(db_name='default', collection_name='default', host='localhost', port=27017, user=None, password=None):
    """
    Simplify getting the client
    """
    if user and password:
        host = "mongodb://%s:%s@%s:%d" % ( quote_plus(user), quote_plus(password), host, port )
        err([host])
        client = MongoClient(host)
    else:
        client = MongoClient(host, port)

    return client

    
def mongo_iterator(db_name='default', collection_name='default', host='localhost', port=27017, user=None, password=None):
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
    verbose = False
    client = get_mongo_client(db_name=db_name, collection_name=collection_name, host=host, port=port, user=user, password=password)
    
    if verbose:
        print("Mongo host:", host, "    port:", port,  "   type:", type(port))
        print("Will iterate on collection:", collection_name)
    
    db = client[db_name]
    coll = db[collection_name]
    f = coll.find()

    return f
    

def mongo_find_one(db_name='default', collection_name='default', host='localhost', port=27017, user=None, password=None):
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
    client = get_mongo_client(db_name=db_name, collection_name=collection_name, host=host, port=port, user=user, password=password)

    if verbose:
        print("Mongo host:", host, "    port:", port,  "   type:", type(port))
        print("Will find one from collection:", collection_name)
    
    db = client[db_name]
    coll = db[collection_name]
    return coll.find_one()


def mongo_count(db_name='default', collection_name='default', host='localhost', port=27017, user=None, password=None):
    """
    Return the number of documents in a collection
    """
    client = get_mongo_client(db_name=db_name, collection_name=collection_name, host=host, port=port, user=user, password=password)
    
    db = client[db_name]
    coll = db[collection_name]

    return coll.count()

    # Alternative method on mothballs:
    n = 0
    iterator = coll.find()
    try:
        for item in iterator:
            n += 1
            if n % 1000 == 0:
                print("n =", n)
    except StopIteration:
        pass
    except Exception as e:
        err([], {'exception':e})
        
    return n


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
