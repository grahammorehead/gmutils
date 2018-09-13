""" neo4j_utils.py

    A set of utils to make Neo4J code simpler

"""
import sys, os, re
import numpy as np
from gmutils.utils import err, argparser, isTrue

from neo4j.v1 import GraphDatabase

################################################################################
# DEFAULTS

################################################################################
# OBJECTS

class HelloWorldExample(object):

    def __init__(self, uri, user, password):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def print_greeting(self, message):
        with self._driver.session() as session:
            greeting = session.write_transaction(self._create_and_return_greeting, message)
            print(greeting)

    @staticmethod
    def _create_and_return_greeting(tx, message):
        result = tx.run("CREATE (a:Greeting) "
                        "SET a.message = $message "
                        "RETURN a.message + ', from node ' + id(a)", message=message)
        return result.single()[0]

    
################################################################################
# FUNCTIONS

def generate_driver(uri, user, passwd):
    """
    Generate the Driver object needed for Neo4J interaction
    """
    driver = GraphDatabase.driver(uri, auth=(user, passwd))
    return driver


def neo4j_graph(query, host, port, user, passwd):
    """
    Open a connection and obtain a graph
    """
    uri = 'bolt://'+ host +':'+ str(port)
    driver = generate_driver(uri, user, passwd)
    with driver.session() as session:
        result = session.run(query)
        return result.graph()


def test(query, host, port, user, passwd):
    """
    Simple test
    """
    G = neo4j_graph(query, host, port, user, passwd)
    print(dir(G))
    
    print("\nNODES:")
    for node in G.nodes:
        print(node)
        
    print("\nRELATIONSHIPS:")
    for relationship in G.relationships:
        rels = []
        for n in relationship.nodes:
            rels.append('(%s)'% list(n.labels)[0])
        rel = '--'.join(rels)
        print('\t', rel)
            
    
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Utils for Neo4J: neo4j_utils.py"})
    parser.add_argument('--query', help='Query for Neo4J', required=False, type=str)
    args = parser.parse_args()

    if args.test:
        test(args.query, args.host, args.port, args.user, args.passwd)
    else:
        print(__doc__)

################################################################################
################################################################################
