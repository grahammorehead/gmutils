""" node.py

Tools to manage nodes in a parse tree

"""
import os, sys, re, json

from gmutils.objects import Object
from gmutils.utils import err, argparser, vector_average


################################################################################
# OBJECTS

class Node(Object):
    """
    A node in a dependency parse tree.  Based on an underlying Spacy Doc

    Attributes
    ----------
    is_dead : boolean
        Used to ensure activity by absorbed nodes

    doc : spacy.Doc

    tokens : array of spacy.Token

    parent : Node

    children : array of Node

    embedding : array of float
        An embedding in some defined vector space

    """
    def __init__(self, spacy_doc, spacy_token, parent=None, options={}):
        """
        Instantiate the object and set options

        spacy_token : spacy.Token

        parent : Node

        """
        self.set_options(options)        # For more on 'self.set_options()' see object.Object
        self.is_dead = False
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


    def get_next_token(self):
        """
        Return the token just after the last token currently held by this Node
        """
        highest = 0
        for token in self.tokens:
            if token.i > highest:
                highest = token.i
        next_index = highest + 1
        if next_index < len(self.doc):
            return self.doc[next_index]
        else:
            return None


    def get_root(self):
        """
        Recursive climb toward the root
        """
        if self.is_root():
            return self
        return self.parent.get_root()


    def node_with_token(self, token):
        """
        Search locally and recursively down to the leaves for the Node containing <token>
        """
        if token in self.tokens:                    # Base case
            return self
        
        for child in self.children:
            node = child.node_with_token(token)     # Recursion
            if node is not None:
                return node
            
        return None
    

    def node_of_token(self, token):
        """
        Given a token, find the Node to which it currently belongs
        """
        root = self.get_root()
        node = root.node_with_token(token)
        return node
        

    def adopt(self, node):
        """
        Create both sides of parent-child relationships
        """
        verbose = False
        
        if isinstance(node, list):
            for n in node:
                self.adopt(n)
        else:
            if node in self.children:
                return
            if node == self:
                return
            if verbose:  print(self, 'adopting', node)
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
        verbose = False
        
        if self == node:  err([self], {'ex':'node = self'})     # Sanity check

        if verbose:
            print('Absorbing', node, 'into', self)
            print('\thaving children:', node.children)
        
        if node in self.children:              # <node> is a child of self
            if verbose:  print(self, 'disowning', node)
            self.disown(node)                  # Separate old child from parent
            self.tokens.extend(node.tokens)    # Absorb their tokens
            self.adopt(node.children)          # Adopt their children
            
        elif self in node.children:            # <node> is the parent of self

            if node.is_root():                 # <node> is root
                print(node, 'is root!')
                exit()
            
            parent = node.parent               # grandparent of self is the new parent
            if verbose:  print(parent, 'disowning', node)
            parent.disown(node)
            self.tokens.extend(node.tokens)    # Absorb their tokens
            self.adopt(node.children)          # Adopt their children
            parent.adopt(self)                 # New parental relationship (sometimes this already exists)
            
        else:
            old_parent = node.parent
            if verbose:  print(old_parent, 'disowning', node)
            old_parent.disown(node)
            self.tokens.extend(node.tokens)    # Absorb their tokens
            self.adopt(node.children)          # Adopt their children
            old_parent.adopt(self)             # New parental relationship (sometimes this already exists)

        node.is_dead = True                # Finally make sure the absorbed Node knows it's been absorbed

        # Final sanity check
        for child in self.children:
            if child == self:
                err([self], {'ex':'child = self'})
            if str(self) == str(child):
                err([self], {'ex':'child = self'})
        
    
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


    def get_entity_type(self):
        """
        Return an array containing the recognized entity types purely for this one entity (No Node should contain more than one).
        Only in the case where spaCy has made an error will this array have more than one element.

        Known types
        -----------
        PERSON
        ORG
        GPE
        CARDINAL
        DATE
        TIME
        MONEY
        LAW

        Returns
        -------
        array of str

        """
        ents = []
        for token in self.tokens:
            if not token.ent_type_ in ents:
                ents.append(token.ent_type_)
        return ents


    def get_entity_position(self):
        """
        If a token is part of an entity, it is either the beginning (B) or the interior (I), which includes the last token.
        "O" refers to being outside any entity.

        Returns
        -------
        str

        """
        position = 'O'
        for token in self.tokens:
            if token.ent_iob_ == 'B':
                position = 'B'
            if token.ent_iob_ == 'I'  and  position == 'O':
                position = 'I'
        return position


    def get_entity_beginners(self):
        """
        Recursively accrue a list of all Nodes representing the beginning of an entity.

        """
        beginners = []
        if self.get_entity_position() == 'B':                   # Base case
            beginners.append(self)

        for child in self.children:
            beginners.extend( child.get_entity_beginners() )    # Recursion

        return beginners

        
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


    def agglomerate_entities(self, options={}):
        """
        For the purpose of dealing sensibly with extracted entities, agglomerate tokens from a single entity into a node.

        No need to apply this function recursively as it is applied separately to all Nodes that represent the beginning
        of an entity.

        """
        altered = False
        if self.is_dead:
            return altered

        if self.get_entity_position() == 'B':             # Base case
            next_token = self.get_next_token()            # Get token immediately following last token of this Node
            next_node = self.node_of_token(next_token)    # Get the Node containing that token
            if next_node.get_entity_position() == 'I':
                self.absorb(next_node)
                altered = True

        return altered


    def agglomerate_verbs_preps(self, vocab=None, options={}):
        """
        For the purpose of sense disambiguation, agglomerate verbs with prepositional children

        e.g. If "jump" is used to describe A jumping over B, the real sense of the verb is "jump over"

        """
        altered = False
        if self.is_dead:
            return altered
            
        if self.is_leaf():
            return altered

        # Select which children, if any, to absorb
        to_absorb = []
        if self.is_verb:
            for child in self.children:
                if 'prep' in child.get_deps():  # Only looking to agglomerate nodes with a 'prep' dependency
                    if vocab is not None:       # Only allow agglomerations corresponding to a vocab entry
                        if not combined_lemmas_in_vocab(vocab, self.get_lemmas_str(), child.get_lemmas_str()):
                            continue
                    to_absorb.append(child)     # Add to the list of children to absorb

        # Execute absorptions in order, re-confirming vocab presence along the way
        for child in to_absorb:
            if combined_lemmas_in_vocab(vocab, self.get_lemmas_str(), child.get_lemmas_str()):
                self.absorb(child)
                altered = True
                    
        # Apply recursively to children
        for child in self.children:
            child.agglomerate_verbs_preps(vocab=vocab)
            
        return altered


    def get_verb_nodes(self):
        """
        From each of the constituent trees, return a list of all nodes that are verbs

        """
        verbs = []
        if self.is_verb():
            verbs = [self]
        
        for child in self.children:
            verbs.extend( child.get_verb_nodes() )
        return verbs


    def embed(self, vocab):
        """
        Given some vocab (embedding) produce a vector that represents this node

        """
        verbose = False

        self.get_entity_type()
        
        # First try full lemma string
        lemmas_str = self.get_lemmas_str()
        lemmas_str = re.sub(r' ', '_', lemmas_str)
        vec = vocab.get(lemmas_str)
        if vec is not None:
            if verbose: print("Found vector for: %s"% lemmas_str)
            self.embedding = vec
            
        elif len(self.get_lemmas()) > 1:   # Use an averaging of as many vectors as available
            if verbose: print("Found nothing for: %s"% lemmas_str)
            vecs = []
            for lemma in self.get_lemmas():
                vec = vocab.get(lemma)
                if vec is not None:
                    if verbose: print("Found sub-vector for: %s"% lemma)
                    vecs.append(vec)
                else:
                    if verbose: print("Found nothing for: %s"% lemma)
            if len(vecs) > 1:
                vec = vector_average(vecs)
            elif len(vecs) == 1:
                vec = vecs[0]
            self.embedding = vec

        # Recursive application
        for child in self.children:
            child.embed(vocab)
            
    
    ############################################################################
    # Printing

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
    

def combined_lemmas_in_vocab(vocab, lemmas_str_A, lemmas_str_B):
    """
    Look for a combination of these two lemmas strings such that vocab has an entry

    """
    lemmas_str = lemmas_str_A +'_'+ lemmas_str_B
    lemmas_str = re.sub(r' ', '_', lemmas_str)
    if vocab.get(lemmas_str) is not None:
        return True
    
    lemmas_str = lemmas_str_B +'_'+ lemmas_str_A
    lemmas_str = re.sub(r' ', '_', lemmas_str)
    if vocab.get(lemmas_str) is not None:
        return True

    return False

        
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
    
