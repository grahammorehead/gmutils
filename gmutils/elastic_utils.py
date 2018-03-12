""" elastic_utils.py

Helper functions for Elasticsearch

"""
import os, sys, re
from editdistance import eval as fast_levenshtein

from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch()

from gmutils.utils import err, argparser, isTrue
from gmutils.normalize import normalize
from gmutils.lexical import damerauLevenshtein, phrase_similarity

################################################################################
# ADMIN FUNCTIONS

def delete_index(index='default'):
    """
    Delete the given index

    """
    sys.stderr.write('Deleting '+ index +'...\n')
    es.indices.delete(index=index, ignore=[400, 404])


def list_indices():
    """
    List all ES indices on a given Elasticsearch server

    """
    I = es.indices.get_alias("*")
    print('\nLocal Elasticsearch indices:')
    for name in I.keys():
        print('\t', name)


################################################################################
# STORING FUNCTIONS        

def store_line(line, index=None, field='name'):
    """
    Convert a line of text into a document and store it in an index as 'field'

    Parameters
    ----------
    line : str
        line to be stored

    index : str
        name of index where things are to be stored

    """
    if index is None:
        index = 'default'
    doc_type = 'word'
    doc = {}
    doc[field] = normalize(line)   # Default: just one word on each line

    es.index(index=index, doc_type=doc_type, body=doc)


def index_dict(datum, index=None):
    """
    Store a dict as a single document in an index

    Parameters
    ----------
    datum : dict
        dict of data to be stored in this record

    index : str
        name of index where things are to be stored

    """
    if index is None:
        index = 'default'
    doc_type = 'dict'
    print("\nDATUM:\n", datum)
    es.index(index=index, doc_type=doc_type, body=datum)


def index_text_with_synonyms(names, index=None):
    """
    Store a list of names as a single document in an index, where each string following the first is a synonym

    Parameters
    ----------
    names : array of str

    index : str
        name of index where things are to be stored

    """
    if index is None:
        index = 'default'

    datum = {
        'name' : names[0],
        'names' : names
        }
    doc_type = 'dict'
    es.index(index=index, doc_type=doc_type, body=datum)


def build_store_action(datum, index='default'):
    """
    For the preparation of storing records via the bulk API

    Parameters
    ----------
    datum : dict

    index : str
        index where the record will be stored

    """
    action = {
        "_index": index,
        "_type": "word",
        '_source': datum
    }
    return action

        
def index_dicts(data, index='default'):
    """
    For a given array of dicts 'data', store each dict as a separate record in an index

    Parameters
    ----------
    data : array of dict

    index : str
        name of index where things are to be stored

    """
    doc_type = 'dict'
    actions = []
    for datum in data:
        actions.append( build_store_action(datum=datum, index=index) )
    
    sys.stderr.write('.')
    helpers.bulk(es, actions)


def execute_individually(actions):
    """
    For a given array of actions, execute each individually (as opposed to using the builk API)

    """
    for action in actions:
        try:
            helpers.bulk(es, [action])
        except:
            pass
    

################################################################################
# SEARCH FUNCTIONS

def word_search(file, index='default'):
    """
    Search for each line of file in the index

    Parameters
    ----------
    index : str
        name of index where things are to be stored

    file : str
        path to file where lines are to be read

    """
    verbose = False
    if file is None:
        err(['word_search requires file'])
        exit()

    seen = Set([])
    iterator = iter_file(file)

    while True:
        try:
            line = iterator.next().rstrip()
            name = None
            try:
                e = line.split('|')
                name = e[0]
            except:
                name = line

            name = normalize(name)
            
            for word in name.split(' '):
                if len(word) < 3:
                    continue
                if word in seen:
                    pass            # only do each one once
                else:
                    seen.add(word)
                    search(word, {'prefix':True})

        except StopIteration:
            break

    
def get_docs(index='default'):
    """
    Parameters
    ----------
    index : str
        name of index where things are stored

    Check in browser:
    http://localhost:9200/default/_search?pretty=true&q=*:*

    """
    docs = []
    res = es.search(index=index, body={"query": {'match_all':{} } })
    print("%d documents found" % res['hits']['total'])
    for doc in res['hits']['hits']:
        docs.append(doc['_source']['name'])
    return docs


def parse_doc_output(doc):
    out = {}
    out['id'] = doc['_id']
    out['score'] = doc['_score']
    out['name'] = doc['_source']['name']
    try:
        out['other_id'] = doc['_source']['other_id']
    except:
        pass

    return out


def match_search(line, index='default'):
    body = {
        "query": {
            "match": {
                "name": {
                    "query":     line,
                    "fuzziness": "AUTO",
                    "operator":  "and"
                }
            }
        }
    }
    res = es.search(index=index, body=body)
    return res['hits']['hits']


def synonym_search(line, index='default'):
    body = {
        "query": {
            "match": {
                "names": {
                    "query":     line,
                    "fuzziness": "AUTO",
                    "operator":  "and"
                }
            }
        }
    }
    res = es.search(index=index, body=body)
    return res['hits']['hits']


def prefix_search(line, index='default'):
    err([line])
    body = {
        "query": {
            "match": {
                "_all": {
                    "query": line,
                    "type": "phrase_prefix",
                    "max_expansions": 100
                }
            }
        }
    }
    res = es.search(index=index, body=body)
    return res['hits']['hits']


def wildcard_search(line, index='default'):
    body = {
             "query": {
               "wildcard": {
                 "name": '*'+ line +'*'
               }
             }
           }
    res = es.search(index=index, body=body)
    return res['hits']['hits']


def search_line(line, index='default', options=None):
    """
    Search for a given substring in an index

    Parameters
    ----------
    index : str
        name of index where things are to be stored

    line : str
        line to be stored

    """
    line = normalize(line)
    docs = []
    seen = Set([])

    for r in match_search(line, index):
        doc = parse_doc_output(r)
        if not doc['id'] in seen:
            seen.add(doc['id'])
            doc['score'] = phrase_similarity(line, doc['name'])
            docs.append(doc)

    if isTrue(options, 'simple'):
        return docs

    for r in prefix_search(line, index):
        doc = parse_doc_output(r)
        if not doc['id'] in seen:
            seen.add(doc['id'])
            doc['score'] = phrase_similarity(line, doc['name'])
            docs.append(doc)
    
    for r in wildcard_search(line, index):
        doc = parse_doc_output(r)
        if not doc['id'] in seen:
            seen.add(doc['id'])
            doc['score'] = phrase_similarity(line, doc['name'])
            docs.append(doc)
    
    docs = sorted(docs, reverse=True, key=lambda x: x['score'])
    assert( isinstance(docs, list) )
    return docs


################################################################################
# MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "Helper functions for Elasticsearch: elastic_utils.py"})

    #  --  Tool-specific command-line args may be added here
    parser.add_argument('--list',            help='List all indices', required=False, action='store_true')
    parser.add_argument('--get',             help='Get all docs from index', required=False, action='store_true')
    parser.add_argument('--delete',          help='Get all docs from index', required=False, action='store_true')
    parser.add_argument('--quiet',           help='Less output to STDOUT, STDERR', required=False, action='store_true')
    
    parser.add_argument('--index',           help='Specify an ES index', required=False, nargs='?', action='append')
    parser.add_argument('--search',          help='Search for this string in ES index', required=False, nargs='?', action='append')
    parser.add_argument('--search_verbose',  help='Search for each line in this file, verbose output', required=False, nargs='?', action='append')
    
    args = parser.parse_args()   # Get inputs and options

    if args.list:
        list_indices()
        exit()

    index = 'default'
    if args.index:
        index = args.index[0]

    if args.delete:
        delete_index(index)
        exit()

    if args.get:
        docs = get_docs(index)
        print('Index', index, 'contains', len(docs), 'documents')
        exit()

    if args.search_verbose:
        search_verbose(args.search_verbose[0], index=index, options=args)
        exit()

    if args.word_search:
        word_search(args.word_search[0], index=index)
        exit()

        
        
################################################################################
