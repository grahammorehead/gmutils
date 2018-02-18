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
        
            
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        return False


    def adopt(self, nodes):
        """
        Create both sides of parent-child relationships

        """
        for node in nodes:
            node.parent = self
            self.children.append(node)
    

    def disown(self, node):
        """
        Break both sides of parent-child relationship

        """
        self.children.remove(node)
        node.parent = None

        
    def absorb(self, node):
        """
        For one Node to be merged with another (usually a parent with a child).
        Afterwards, nothing should link to 'node'

        """
        self.disown(node)
        self.adopt(node.children)
        self.tokens.extend(node.tokens)
        
    
    def get_text(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        texts = []
        for token in self.tokens:
            texts.append(token.text)
        return ' '.join(texts)


    def get_lemmas(self):
        """
        Get a string representing, in tree order, the tokens comprising this Node

        """
        lemmas = []
        for token in self.tokens:
            lemmas.append(token.lemma_)
        return lemmas


    def get_lemmas_str(self):
        return ' '.join(self.get_lemmas())


    def get_pos(self):
        """
        Get the part of speech (could be multiple)

        """
        pos = []
        for token in self.tokens:
            pos.append(token.pos_)
        return pos


    def is_verb(self):
        if 'VERB' in self.get_pos():
            return True
        return False
    
    
    def get_deps(self):
        """
        Get the dependency relation type (could be multiple)

        """
        deps = []
        for token in self.tokens:
            deps.append(token.dep_)
        return sorted(deps)


    def get_deps_str(self):
        """
        A simple str representation of the dependency type

        """
        return ' ' .join(self.get_deps())
    
    
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
    

    def get_semantic_roles(self, options={}):
        """
        Glean what we can about semantic roles if self is_verb

        Returns
        -------
        dict
          dep_type : list of spacy.Token
          Maps the dependency type onto an array of Tokens and therefore a substring

        """
        roles = {}
        for child in self.children:
            deps = child.get_deps_str()

            if deps == 'punct'  and  child.is_leaf():
                continue
            
            if deps in roles:
                roles[deps].append(child)
            else:
                roles[deps] = [child]

        return roles


    def agglomerate_verbs_preps(self, options={}):
        """
        For the purpose of sense disambiguation, agglomerate verbs with prepositional children

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"

        """
        if self.is_leaf():
            return
        
        if self.is_verb:
            for child in self.children:
                if 'prep' in child.get_deps():
                    self.absorb(child)
        
        for child in self.children:
            child.agglomerate_verbs_preps()
            

    def print_semantic_roles(self, options={}):
        """
        Print what we know about the semantic roles

        """
        print("\nSemantic Roles:")
        print(self.get_lemmas_str())
        roles = self.get_semantic_roles(options=options)
        for dep, nodes in roles.items():
            print('    {%s}'% dep, get_support_for_nodes(nodes))
    
            
    def pretty_print(self, depth=0, options={}):
        """
        Print out the tree recursively from this point

        """
        indent = depth * '    '
        print(indent + self.get_text() + ' {%s}'% self.get_deps_str())

        # Options
        if options.get('supporting text'):  # Print the text supporting subtree
            print(indent + '[ST]: ' + self.get_supporting_text())
        
        # Recursion
        for child in self.children:
            child.pretty_print(depth + 1, options=options)

        if self.is_root():
            self.print_semantic_roles(options=options)
            

            
################################################################################
# FUNCTIONS

def iprint(title, X):
    first = True
    for x in X:
        if first:
            print(' - ' + title)
            first = False
        print('\t', x)
        

def get_support_for_tokens(doc, tokens):
    """
    Give a doc and some tokens, find the shortest contiguous substring containing all the tokens

    """
    if len(tokens) == 0:
        return ''
    
    left = right = None
    for token in tokens:
        if left is None or token.left_edge.i < left:
            left = token.left_edge.i
        if right is None or token.right_edge.i > right:
            right = token.right_edge.i

    subtree_span = doc[left:right + 1]
    return subtree_span.text
    

def get_support_for_nodes(nodes):
    """
    For some nodes (from single Document), find the shortest contiguous substring containing all constituents tokens

    """
    if len(nodes) == 0:
        return ''
    
    doc = left = right = None
    for node in nodes:
        if doc is None:
            doc = node.doc
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
    
