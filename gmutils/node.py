""" node.py

Tools to manage nodes in a parse tree

"""
import os, sys, re, json

from gmutils.objects import Object
from gmutils.utils import err, argparser


################################################################################
# OBJECTS

class Node(Object):
    """
    A node in a dependency parse tree.  Based on an underlying Spacy Doc

    Attributes
    ----------
    doc : spacy.Doc

    tokens : array of spacy.Token

    parent : Node

    children : array of Node

    """
    def __init__(self, spacy_doc, spacy_token, parent=None, options={}):
        """
        Instantiate the object and set options

        spacy_token : spacy.Token

        parent : Node

        """
        self.set_options(options)        # For more on 'self.set_options()' see object.Object
        self.doc = spacy_doc

        if isinstance(spacy_token, list):
            self.tokens = spacy_token
        else:
            self.tokens = [spacy_token]
        
        self.parent = parent
        self.children = []
        
        # Base original tree on spacy Token tree
        for token in self.tokens:
            for child in token.children:
                self.children.append(Node(self.doc, child, parent=self, options=options))


    def __repr__(self):
        return self.get_text()

        
    def is_root(self):
        if self.parent is None:
            return True
        return False
        
            
    def get_text(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        texts = []
        for token in self.tokens:
            texts.append(token.text)
            
        return ' '.join(texts)


    def get_lemma(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        lemmas = []
        for token in self.tokens:
            lemmas.append(token.lemma_)
            
        return ' '.join(lemmas)


    def get_dep(self):
        """
        Get the dependency relation type

        """
        deps = []
        for token in self.tokens:
            deps.append(token.dep_)

        return deps


    def dep_str(self):
        """
        A simple str representation of the dependency type

        """
        return ' ' .join(self.get_dep())
    
    
    def get_supporting_text(self):
        """
        Find the shortest substring in the original text such that all nodes on this subtree are represented

        """
        left = right = None
        for token in self.tokens:
            if left is None or token.left_edge.i < left:
                left = token.left_edge.i
            if right is None or token.right_edge.i > right:
                right = token.right_edge.i
                
        subtree_span = self.doc[left:right + 1]
        return subtree_span.text
    

    def pretty_print(self, depth=0, options={}):
        """
        Print out the tree recursively from this point

        """
        indent = depth * '    '
        print(indent + self.get_text() + ' {%s}'% self.dep_str())

        # Options
        if options.get('supporting text'):  # Print the text supporting subtree
            print(indent + '[ST]: ' + self.get_supporting_text())
        
        # Recursion
        for child in self.children:
            child.pretty_print(depth + 1, options=options)

        if self.is_root():
            self.print_semantic_roles(options=options)
            

    def print_semantic_roles(self, options={})
        """
        Print what little we know about the semantic roles

        """
        # Semantic roles for root verb
        print("\nSR: %s"% self.get_lemma())
        for child in self.children:
            if 'prep' in child.get_dep():
                print('    {%s} [%s] '% (child.dep_str(), child.get_lemma()), get_support_for_nodes(self.doc, child.children))
            else:
                print('    {%s} '% child.dep_str() + get_support_for_nodes(self.doc, [child]))
        print()


    
            
            
################################################################################
# FUNCTIONS

def iprint(title, X):
    first = True
    for x in X:
        if first:
            print(' - ' + title)
            first = False
        print('\t', x)
        

def get_support_for_nodes(doc, nodes):
    """
    Give a doc and some nodes, find the shortest contiguous substring containing all constituents tokens

    """
    if len(nodes) == 0:
        return ''
    
    left = right = None
    for node in nodes:
        for token in node.tokens:
            if left is None or token.left_edge.i < left:
                left = token.left_edge.i
            if right is None or token.right_edge.i > right:
                right = token.right_edge.i

    subtree_span = doc[left:right + 1]
    return subtree_span.text
    

        
################################################################################
# MAIN

if __name__ == '__main__':
    parser = argparser({'desc': "Tools to handle dependency parses: node.py"})
    args = parser.parse_args()   # Get inputs and options

    if args.str:
        pass
    else:
        print(__doc__)

        
################################################################################
################################################################################
    
