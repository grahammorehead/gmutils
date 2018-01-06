""" elastic_utils.py

Helper functions for Elasticsearch

"""
import os, sys, re
from sets import Set
from editdistance import eval as fast_levenshtein

from elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch()

from .utils import *
from .normalize import normalize
from .lexical import damerauLevenshtein, phrase_similarity


def list_indices():
    """
    List all ES indices
    """
    I = es.indices.get_alias("*")
    print('\nLocal Elasticsearch indices:')
    for name in I.keys():
        print('\t', name)


def build_doc_from_line(line):
    """
    Takes str delimited by "|", builds dict
    """
    out = {}
    e = line.split("|")

    # Default: just one word on each line
    out['name'] = normalize(line)

    return out
    

def store_line(line, index=None):
    """
    Parameters
    ----------
    index : str
        name of index where things are to be stored

    line : str
        line to be stored

    """
    if index is None:
        index = 'default'

    doc_type = 'word'
    doc = build_doc_from_line(line)
    
    es.index(index=index, doc_type=doc_type, body=doc)
    # print('index=', index, 'doc_type=', doc_type, 'body=', doc)


def build_action_from_line(index='default', line=None):
    action = {
        "_index": index,
        "_type": "word",
        '_source': {
            "name": line
        }
    }
    return action

        
def build_action(index='default', name=None, rfc=None):
    action = {
        "_index": index,
        "_type": "word",
        '_source': {
            "name": name,
            "rfc": rfc
        }
    }
    return action

        
def has_invalid_chars(line):
    for a in list(line):
        if re.search(u"[&A-Z0-9'\-\xd1\.\, ]", a):
            pass
        else:
            # print("INVALID: ", [line])
            return True
    return False
        

def store_lines(lines, index='default'):
    """
    Parameters
    ----------
    index : str
        name of index where things are to be stored

    lines : array of str
        lines to be stored

    """
    doc_type = 'word'
    actions = []
    for line in lines:
        line = normalize(line)
        line = clean_for_deab(line)
        
        if not has_invalid_chars(line):
            actions.append( build_action_from_line(index=index, line=line) )
    
    sys.stderr.write('.')
    helpers.bulk(es, actions)


def store_actions_individually(actions):
    for action in actions:
        try:
            helpers.bulk(es, [action])
        except:
            pass
    

def store_file(file=None, index='default'):
    """
    Parameters
    ----------
    index : str
        name of index where things are to be stored

    file : str
        path to file where lines are to be read

    """
    if file is None:
        err(['store_file requires file'])
        exit()

    batch = 999
    iterator = iter_file(file)
    actions = []
    
    while True:
        try:
            line = iterator.next().rstrip().upper()
            if re.search('RAZON_SOCI', line):
                continue

            name = rfc = None
            try:
                e = line.split('|')
                name = normalize(e[0])
                rfc  = e[1]
                rfc = unicode(clean_for_deab(rfc))
            except:
                name = line
            name = clean_for_deab(name)
            
            action =  build_action(index=index, name=name, rfc=rfc)
            actions.append(action)
            if len(actions) > batch:
                sys.stderr.write('.')
                try:
                    helpers.bulk(es, actions)
                except:
                    store_actions_individually(actions)
                actions = []
            
        except StopIteration:
            break
        
    
def store_words(file=None, index='default'):
    """
    Parameters
    ----------
    index : str
        name of index where things are to be stored

    file : str
        path to file where lines are to be read

    """
    if file is None:
        err(['store_file requires file'])
        exit()

    batch = 1000   # size of actions to be performed in bulk
        
    seen = {}
    iterator = iter_file(file)
    lines = []

    while True:
        try:
            line = iterator.next().rstrip().upper()
            if re.search('RAZON_SOCI', line):
                continue

            name = rfc = None
            try:
                e = line.split('|')
                name = e[0]
            except:
                name = line
            name = clean_for_deab(name)
            
            for word in name.split(' '):
                if word in seen:
                    seen[word] += 1
                    if seen[word] == 5:
                        lines.append(word)
                        if len(lines) >= batch:
                            store_lines(lines)
                            lines = []
                else:
                    seen[word] = 1

        except StopIteration:
            break


def word_search(file, index='default'):
    """
    Parameters
    ----------
    index : str
        name of index where things are to be stored

    file : str
        path to file where lines are to be read

    """
    verbose = False
    if file is None:
        err(['store_file requires file'])
        exit()

    seen = Set([])
    iterator = iter_file(file)

    while True:
        try:
            line = iterator.next().rstrip().upper()
            if re.search('RAZON_SOCI', line):
                continue

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
        name of index where things are to be stored

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
        out['rfc'] = doc['_source']['rfc']
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


def prefix_search(line, index='default'):
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


def search_company(name, options=None):
    """
    Search for <name> in the 'universe' index
    """
    docs = search_line(name, 'universe', options)
    if docs:
        if docs[0]['score'] > 0.6:
            return docs[0]
        
    return None

    
def delete_index(index='default'):
    sys.stderr.write('Deleting '+ index +'...\n')
    es.indices.delete(index=index, ignore=[400, 404])


################################################################################
##   MAIN   ##

if __name__ == '__main__':

    parser = argparser({'desc': "Helper functions for Elasticsearch: elastic_utils.py"})

    #  --  Tool-specific command-line args may be added here
    parser.add_argument('--list',  help='List all indices', required=False, action='store_true')
    parser.add_argument('--get',   help='Get all docs from index', required=False, action='store_true')
    parser.add_argument('--delete',   help='Get all docs from index', required=False, action='store_true')
    parser.add_argument('--quiet',   help='Less output to STDOUT, STDERR', required=False, action='store_true')
    
    parser.add_argument('--index',  nargs='?', action='append', help='Specify an ES index', required=False)
    parser.add_argument('--store',  nargs='?', action='append', help='Store each line of file in ES index', required=False)
    parser.add_argument('--store_words',  nargs='?', action='append', help='Store each word of each line of file in ES index', required=False)
    parser.add_argument('--search',  nargs='?', action='append', help='Search for this string in ES index', required=False)
    parser.add_argument('--search_company',  nargs='?', action='append', help='Search for this string in the ES index', required=False)
    parser.add_argument('--search_company_substring',  nargs='?', action='append', help='Search for this substring in the ES index', required=False)
    parser.add_argument('--word_search',  nargs='?', action='append', help='Search each line of file in ES index', required=False)
    parser.add_argument('--search_verbose',  nargs='?', action='append', help='Search for each line in this file, verbose output', required=False)
    parser.add_argument('--study',  nargs='?', action='append', help='A value to help study the results of the search (only some tasks)', required=False)

    
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
        for doc in get_docs(index):
            print('\t', doc)
        exit()

    if args.store:
        store_file(file=args.store[0], index=index)
        exit()
        
    if args.store_words:
        store_words(file=args.store_words[0], index=index)
        exit()
        
    if args.search:
        for doc in search_line(args.search[0], index=index):
            print(doc['score'], ':', doc['rfc'], '|', doc['name'])
        exit()

    if args.search_company:
        query = args.search_company[0]
        doc = search_company(query)
        if doc is None:
            print('\nNo Results.\n')
            exit()
            
        print('\nQuery: "%s"'% query)
        print('\tTop Score:  %0.3f'% doc['score'])
        print('\tRFC:        %s'% doc['rfc'])
        print('\tName:       %s\n'% doc['name'])
        exit()

    if args.search_company_substring:
        docs = search_line(args.search_company_substring[0], index='universe')
        if not docs:
            print('\nNo Results.\n')
            exit()
        for doc in docs:
            print(doc['score'], ' : ', doc['rfc'], ' | ', doc['name'])
        exit()

    if args.search_verbose:
        search_verbose(args.search_verbose[0], index=index, options=args)
        exit()

    if args.word_search:
        word_search(args.word_search[0], index=index)
        exit()

        
        
################################################################################
