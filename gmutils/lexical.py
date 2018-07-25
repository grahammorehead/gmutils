""" lexical.py

    Some Tools to assist with lexical operations

"""
import time, sys, re
from numpy import sqrt
from editdistance import eval as fast_levenshtein
from gmutils.utils import err
from gmutils.normalize import simplify_for_distance

################################################################################
# DEFAULTS

low_val_str = "i, me, my, myself, we, our, ours, ourselves, you, your, yours, yourself, yourselves, he, him, his, himself, she, her, hers, herself, it, its, itself, they, them, their, theirs, themselves, what, which, who, whom, this, that, these, those, am, is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and, but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through, during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further, then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some, such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, should, now"

low_val_words = set(low_val_str.split(', '))

medium_val_words = set([])

################################################################################

def letterCost(a, b):
    if a==b:
        return 0
    else:
        return 1


def letterCost_suppress_vowels(a, b):
    verbose = False
        
    cost = 1.0
    if re.search(r'[AEIOUYaeiouy]', a)  and  re.search(r'[AEIOUYaeiouy]', b):
        cost = 0.5
            
    if verbose:
        err([a, b, cost])
        
    if a==b:
        return 0
    else:
        return cost


def damerauTranspose(s, t):
    '''
    Will effectuate transposes if it promises to lower the Levenshtein cost.  For each possible
    transposition only two possibilities will be considered.  Most cases should be covered.

    Transpositions alone would implement Damerau-Levenshtein.  This goes just a little further
    to attempt to better approximate the triangle inequality
    '''
    S = set(list(s))
    T = set(list(t))
    I = S.intersection(T)
    cost = 0  # cost will increase with each transposition

    # Find subsets of each string that match intersection
    interS, interT = [], []
    for j in range(len(s)):
        if s[j] in I:
            interS.append(j)

    for i in range(len(t)):
        if t[i] in I:
            interT.append(i)

    # iterate through substrings looking for transpositions
    for k in range(len(interS)):
        j  = interS[k]
        i, i1, i2 = None, None, None
        try:
            i  = interT[k]
            if s[j] == t[i]:  continue
        except:  break

        try:
            i1 = interT[k+1]
            if s[j] == t[i1]:
                t = swapIndices(t, i, i1)
                cost += 1
                continue
        except:  pass

        try:
            i2 = interT[k+2]
            if s[j] == t[i2]:
                t = swapIndices(t, i, i2)
                cost += 1
        except:  pass

    return s, t, cost


def print_lev_matrix(A, B, D):
    """
    Print out a levenshtein matrix
    """

    # Print top row
    print('\n\t', end='')
    for a in A:
        print('%s\t'% a, end='')
    print
    
    # Iterate other rows
    high_i = 0
    high_j = 0
    for i in range(len(B)):
        
        b = B[i]
        print('\n%s\t'% b, end='')

        high_i = i
        
        # Iterate columns
        for j in range(len(A)):
            
            d = D[i][j]
            print('%s\t'% d, end='')
            
            high_j = j

    err([high_i, high_j, D[high_i][high_j]])

            
def levenshtein(s, t, cost=letterCost, options={}):
    ''' 
    For two strings s,t will calculate a generalization of the Levenshtein distance function.  For
    two non-strings, a cost function should be provided.  The algorithm below is formulated to work
    well with levenshtein_phrase()
    '''
    # NOTE:
    #     s,n,j are associated
    #     t,m,i are associated
    
    verbose = False
    if verbose:
        print( '\n\nComparing:  ',s,':',t )

    # length() that handles null objects
    def length(x):
        if x == None:
            return 0
        try:
            return len(x)
        except:
            return 1
        return 1
    
    # Base Cases
    n = length(s)
    m = length(t)
    nf = max(m,n,1)
    if n == 0:
        if options.has_key('normFactor'):
            return m, nf
        return m
    if m == 0:
        if options.has_key('normFactor'):
            return n, nf
        return n
    
    # Build initial Levenshtein matrix d[row][col]
    d = []
    norm = []   # parallel table to hold normalization factors
    for i in range(m+1):
        d.extend([[0]*(n+1)])
        norm.extend([[0]*(n+1)])
    d[0][0] = 0
    norm[0][0] = 0
    space = 0
    if type(s) == type([]):
        space = 1

    for j in range(1,n+1):
        inc = length(s[j-1])
        d[0][j] = d[0][j-1] + inc
        norm[0][j] = norm[0][j-1] + inc
        if j > 1:
            norm[0][j] += space    # Account for spaces

    for i in range(1,m+1):
        inc = length(t[i-1])
        d[i][0] = d[i-1][0] + inc
        norm[i][0] = norm[i-1][0] + inc
        if i > 1:
            norm[i][0] += space    # Account for spaces

    for i in range(1, m+1):
        for j in range(1, n+1):
            incI = length(s[j-1])
            incJ = length(t[i-1])
            norm[i][j] = max(norm[i-1][j], norm[i][j-1])

    # Assign value to each cell in d-table
    firstCost = 0.0
    for i in range(1, m+1):
        for j in range(1, n+1):

            a,b = None,None   # initialize vars to be used in cost function
            try:  a = s[j-1]  # assign s[j-1] if it exists
            except: pass
            try:  b = t[i-1]  # assign t[i-1] if it exists
            except: pass

            this_cost = cost(a,b)
            # Ascribe more importance to first letter comparisons
            if i==1 and j==1:
                first_cost = this_cost

            # Choose the min from 4 options, one of which is the "norm", or length required
            #   to make longer sequence(up to that point) from scratch.  This is considered
            #   the longest theoretical difference for our purposes
            d[i][j] = min( d[i-1][j] + len(t[i-1]),
                           d[i][j-1] + len(s[j-1]),
                           d[i-1][j-1] + this_cost,
                           norm[i][j] )
                
    # Print Levenshtein matrices to analyze function
    if verbose:
        print_lev_matrix(s, t, d)
        
    final_cost = d[m][n]
    if verbose:
        err([m, n, final_cost])
        
    if 'first important' in options:
        if options['first important']:
            final_cost += first_cost
            
    if options.has_key('normFactor'):
        return [final_cost, norm[m][n]]

    return final_cost


def levenshtein_r(s, t, cost=letterCost):
    ''' Recursively calculate the levenshtein distance between arbitrary sequences s and t

    Much much slower than the iterative function above, but can be used to confirm results in testing
    '''
    def total(x):
        tot = 0
        for e in x:
            tot += len(e)
        return tot

    # NOTE:
    #     s,n,j are associated
    #     t,m,i are associated
    n = len(s)
    m = len(t)
    l1, l2 = None, None
    if n == 0:
        return total(t)
    if m == 0:
        return total(s)
    if m == n == 1:
        s = s[0]
        t = t[0]
        return min(len(s)+len(t), cost(s, t))
    else:
        l1 = levenshtein_r(s[:-1], t, cost)                # d[i-1][j] + di,
        l2 = levenshtein_r(s, t[:-1], cost)                # d[i][j-1] + dj,
        l3 = levenshtein_r(s[:-1], t[:-1], cost)           # d[i-1][j-1] + cost(a,b)
        return min(l1+len(s[-1]), l2+len(t[-1]), l3+cost(s[-1], t[-1]))


def damerauLevenshtein(s, t, options={}):
    """
    Finds the lowest of Damerau/Levenstein distance functions
    """
    verbose = False
    if verbose:
        err([s, t])
    maxL = max(len(s), len(t))
    minL = min(len(s), len(t))

    if options.get('ignore_case'):
        s = s.lower()
        t = t.lower()
    
    if options.get('abridge'):  # Cut both strings to same length
        s = s[:minL]
        t = t[:minL]
        
    l = fast_levenshtein(s, t)

    # transpose letters in an attempt to lower Levenshtein cost, each transposition has a cost of 1
    # tc = transposition cost
    s, t, tc = damerauTranspose(s, t)

    d = l
    if tc > 0:
        d = fast_levenshtein(s, t)
        d += tc

    dist = min(l, d)

    if options.get('fancy'):          # Fancy adjustments
        if dist < maxL:
            
            # Higher cost for first letter
            if s[0].lower() != t[0].lower():
                dist += 0.3 * (maxL - dist)

    if options.get('ratio'):
        return dist / maxL
                
    return dist


def damerauLevenshtein_norm(s, t, cost=letterCost):
    ''' Finds the lowest of Damerau/Levenstein distance functions, returns normalized result '''
    l, n = levenshtein(s, t, cost, {'normFactor':True})

    # transpose letters in an attempt to lower Levenshtein cost, each transposition has a cost of 1
    s, t, tc = damerauTranspose(s, t)
    d = tc + levenshtein(s, t, cost)

    dist =  min(l, d)

    return dist/n


def damerauLevenshtein_r(s, t, cost=letterCost):
    l = levenshtein_r(s, t, cost)

    # transpose letters in an attempt to lower Levenshtein cost, each transpose has a cost of 1
    s, t, tc = damerauTranspose(s, t)
    d = tc + levenshtein_r(s, t, cost)

    return min(l, d)


def levenshtein_phrase(s, t):
    ''' 
    For two multi-word strings, will calculate a generalization of the Levenshtein distance between
    the two lists, where each word is treated as a separate element, and the cost function between
    any two strings is the usual levenshtein distance.
    '''
    s.strip()
    t.strip()
    a = s.split(' ')
    b = t.split(' ')
    return levenshtein(a, b, levenshtein)


def damerauLevenshtein_strings(s, t):
    ''' 
    For two lists of strings, will calculate a generalization of the Damerau-Levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the damerau-levenshtein distance.
    '''
    return damerauLevenshtein(s, t, damerauLevenshtein)


def damerauLevenshtein_phrase(s, t):
    ''' 
    For two multi-word strings, will calculate a generalization of the Damerau-Levenshtein distance
    between the two lists, where each word is treated as a separate element, and the cost function
    between any two strings is the damerau-levenshtein distance.
    '''
    s.strip()
    t.strip()
    a = s.split(' ')
    b = t.split(' ')

    return damerauLevenshtein_strings(a, b)


def damerauLevenshtein_strings_norm(s, t):
    ''' 
    For two lists of strings, will calculate a generalization of the Damerau-Levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the damerau-levenshtein distance.  The final cost is
    normalized based on the greatest possible cost
    '''
    d, n = damerauLevenshtein(s, t, damerauLevenshtein, {'normFactor':True})

    # return normalized distance
    result = d/n
    if result > 1:
        print ("\nERROR: result",result,"> 1")
        print ('s:',s)
        print ('t:',t)
        exit()

    return result


def levenshtein_strings_norm(s, t):
    ''' 
    For two lists of strings, will calculate a generalization of the Levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the levenshtein distance.  The final cost is
    normalized based on the greatest possible cost
    '''
    d, n = levenshtein(s, t, levenshtein, {'normFactor':True})

    # return normalized distance
    result = d/n
    if result > 1:
        print ("\nERROR: result",result,"> 1")
        print ('s:',s)
        print ('t:',t)
        exit()

    return result


def damerauLevenshtein_phrase_norm(s, t):
    ''' 
    For two multi-word strings, will calculate a generalization of the normalized Damerau-Levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the typical damerau-levenshtein distance.
    '''
    s.strip()
    t.strip()
    a = s.split()
    b = t.split()

    return damerauLevenshtein_strings_norm(a, b)


def damerauLevenshtein_phrase_r(s, t):
    ''' 
    For two multi-word strings, will RECURSIVELY calculate a generalization of the Damerau-Levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the damerau-levenshtein distance.
    '''
    s.strip()
    t.strip()
    a = s.split(' ')
    b = t.split(' ')
    return damerauLevenshtein_r(a, b, damerauLevenshtein_r)


def levenshtein_phrase_r(s, t):
    ''' 
    For two multi-word strings, will RECURSIVELY calculate a generalization of the levenshtein
    distance between the two lists, where each word is treated as a separate element, and the cost
    function between any two strings is the usual levenshtein distance.
    '''
    s.strip()
    t.strip()
    a = s.split(' ')
    b = t.split(' ')
    return levenshtein_r(a, b, levenshtein_r)


def swapIndices(s, i, j):
    ''' swaps the index of the two elements in s '''
    if i == j: return s

    if isinstance(s, str):
        if i < j:
            a,b = i,j
        else:
            a,b = j,i
        s = s[:a]+s[b]+s[a+1:b]+s[a]+s[b+1:]
    else:
        s[i], s[j] = s[j], s[i]

    return s


def stableSubsequences(seq, k=1):
    ''' Find all subsequences of at least <k> elements in the arbitrary list <seq>.  The output is "stable"
    in that elements stay in their original relative order '''
    out = []
    n = len(seq)
    maxB = (2**n) - 1
    b = 1
    while b <= maxB:
        subSeq = []
        binary = int2binary(b, n)   # use next binary number to know which elements to include
        for i in range(len(binary)):
            if binary[i] == '1':
                subSeq.append(seq[i])
        b += 1
        if len(subSeq) >= k:
            out.append(subSeq)
    return out
    

def stableSubstrings(s, k=1):
    ''' Find all substrings with at least <k> words in the string <s>.  The output is "stable" in that
    words stay in their original relative order '''
    s = removeUnneededSpaces(s)
    seq = s.split(" ")
    out = []
    for o in stableSubsequences(seq, k):
        out.append(" ".join(o))
    return out


def removeUnmatchedStrings(guide, seq, thresh=.4):
    ''' <guide>, <seq> are both lists of strings.  Returns a stable subsequence consisting of the
    strings in <seq> that had a damerauLevenshtein_norm of at least <thresh> with at least one
    string in <guide>
    '''
    out = []
    for s in seq:
        for g in guide:
            if damerauLevenshtein_norm(s,g) <= thresh:
                out.append(s)
                break
    return out


def removeUnmatchedWords(guide, string, thresh=.66):
    ''' <guide>, <seq> are both strings.  Returns a stable substring consisting of the
    words in <seq> that had a damerauLevenshtein_norm of at least <thresh> with at least one
    word in <guide>
    '''
    out = []
    guide = removeUnneededSpaces(guide)
    g = guide.split(" ")
    string = removeUnneededSpaces(string)
    s = string.split(" ")
    return ' '.join(removeUnmatchedStrings(guide, seq, thresh))
        

def cmp_len(x, y):
        ''' compare function based on list length '''
        if len(x) < len(y):
            return -1
        if len(x) > len(y):
            return 1
        else:
            return 0


def removeSubphrasesFromList(list):
    ''' <list>=a list of strings. Removes elements that are subphrases of another element  '''
    listOfLists = []
    for e in list:
        listOfLists.append(e.split(' '))

    longestFirst = sorted(listOfLists, cmp_len, reverse=True)

    final = []
    seen = {}
    num_strings = len(longestFirst)

    for i in range(num_strings):
        length = len(longestFirst[i])
        full = ' '.join(longestFirst[i])
        if not seen.has_key(full):
            final.append(full)
            seen[full] = True

        # inhibit subphrases
        for j in range(length)[1:]:
            partial = ' '.join(longestFirst[i][j:])
            if not seen.has_key(partial):
                seen[partial] = True
            partial = ' '.join(longestFirst[i][:length-j])
            if not seen.has_key(partial):
                seen[partial] = True

    return final

      
def stringDissimilarToStringlist(s, list):
    closeEnough = .4

    for t in list:
        if damerauLevenshtein_strings_norm(s, t) < closeEnough:
            return False
    return True


def ngrams(list):
    ''' Takes a list of words, returns a list of all unigrams, bigrams, and trigrams '''
    out = []
    for l in range(1,4):
        for i in range(len(list)-l):
            out.append(list[i:i+l])
    return out


def string_distance(A, B, options={}):
    """
    Normalized Levenshtein-Damerau distance between two strings
    """
    verbose = False
    len_A = len(A)
    len_B = len(B)
    length = float( max( len_A, len_B ) )
    abs_dist = damerauLevenshtein(A, B)
    
    # diff_length = abs(len_A - len_B)
    # rel_diff_length = diff_length / length
    
    distance = abs_dist / length
    if verbose:
        err([A, B, abs_dist, length, distance])
    
    return distance
    

def find_and_rm_perfect_match(token, indices, A):
    """
    for a given token, will search through the indices in A for the best match in A
    """
    verbose = False
    
    keep = []
    best_i = None

    for i in indices:
        if token == A[i]:
            best_i = i
            
    if verbose:
        err([token, A, indices])
        err([best_i])
        err([A[best_i]])
            
    try:
        if best_i is not None:
            indices.remove(best_i)
    except:
        pass
    return best_i, indices


def find_and_rm_best_match(token, indices, A, options={}):
    """
    for a given token, will search through the indices in A for the best match in A
    """
    verbose = False
    if len(indices) < 1:
        err(["ERROR: empty list of indices"])
        exit()
    
    keep = []
    smallest_distance = 1.0    # distance goes between [0,1]
    best_i = None

    for i in indices:
        distance = string_distance(token, A[i], {'suppress vowels':True})
        if distance > 1.0:
            err([token, A[i]])
            exit()

        if distance <= smallest_distance:
            smallest_distance = distance
            if distance < 0.7:
                best_i = i

    if best_i is not None:
        if options.get('boost_long_words'):    # Special accounting for longer, i.e. more meaningful words
            if len(token) >= 4:
                if len(A[best_i]) >= 4:
                    smallest_distance = sqrt(smallest_distance)

        if verbose:
            err([token, A, indices])
            err([best_i, smallest_distance])
            err([A[best_i]])
        try:
            indices.remove(best_i)
        except:
            pass
        
    return smallest_distance, best_i, indices


def process_perfect_matches(A, B):
    """
    Look for words that match between A, B and perform some overhead

    Parameters
    ----------
    A : array of str

    B : array of str

    Returns
    -------
    closest : dict of int
        matches each token with its best match if one exists

    indices_A : array of int
        remaining unmatched indices of tokens in A

    indices_B : array of int
        remaining unmatched indices of tokens in B
    """
    verbose = False
    
    indices_A = list(range(len(A)))   # track remaining unmatched indices of tokens in A
    indices_B = list(range(len(B)))   # track remaining unmatched indices of tokens in B
    closest = {}
    #  closest = Memory of closest matches for re-order
    #                    key: index in A
    #                    val: index in B

    # Iterate over tokens in B
    for i_B,token_B in enumerate(B):
        if indices_A:
            i_A, indices_A = find_and_rm_perfect_match(token_B,indices_A, A)
            if i_A is not None:
                closest[i_A] = i_B       # Store match using indices
                indices_B.remove(i_B)
                if verbose:
                    token_A = A[i_A]
                    print('\tPerfect match  A:%s  ~  B:%s'% (token_A, token_B))

    return closest, indices_A, indices_B


def marginal_cost(i_A, token_A, i_B, token_B, distance=1.0):
    """
    Associate a cost with this distance.  Uses two external word lists
    """
    verbose = False
    
    # Both words can be safely ignored
    if  token_A in low_val_words  and  token_B in low_val_words:
        return 0.0

    # One word may be ignored
    if  token_A in low_val_words  or  token_B in low_val_words:
        return 0.25 * distance

    # If either word may be semi-ignored
    if token_A in medium_val_words  or  token_B in medium_val_words:
        return 0.5 * distance     # Edit distance cost for medium value words

    # Pay more attention to first word
    if  i_A==0  or  i_B==0:
        rem_dist = 1.0 - distance
        return distance + 0.5*rem_dist

    return distance


def process_best_matches(A, B, closest, indices_A, indices_B):
    """
    Process best remaining matches between A, B.  Starts where 'perfect_matches' leaves off.  Handle some overhead.

    cost calculations begin here

    Parameters
    ----------
    A : array of str

    B : array of str

    closest : dict of int
        matches each token with its best match if one exists

    indices_A : array of int
        remaining unmatched indices of tokens in A

    indices_B : array of int
        remaining unmatched indices of tokens in B

    Returns
    -------
    closest : dict of int
        matches each token with its best match if one exists

    indices_A : array of int
        remaining unmatched indices of tokens in A

    indices_B : array of int
        remaining unmatched indices of tokens in B
    """
    verbose = False
    cost = 0.0
    
    # Iterate over remaining tokens in B (filtered on indices_B)
    #     from longest to shortest, finding best matches.
    #     Keep track of which tokens in B (if any) don't have a match.
    unmatched_i_B = []
    for i_B,token_B in sorted( enumerate(B), reverse=True, key=lambda x: len(x[1])):
        if i_B not in indices_B:
            continue

        # If there are indices in A that remain unmatched
        if indices_A:

            # Attempt to find a match for token_B
            distance, i_A, indices_A = find_and_rm_best_match(token_B,indices_A, A)
            if distance > .21:
                unmatched_i_B.append(i_B)
            if verbose:
                err([distance, i_A, indices_A])
                
            if i_A is None:  # no suitable match was found
                continue

            # A match was found
            closest[i_A] = i_B           # Store match using indices
            token_A = A[i_A]
            if verbose:
                print('\tBest match (%0.4f)  A:%s  ~  B:%s'% (distance, token_A, token_B))

            cost += marginal_cost(i_A, token_A, i_B, token_B, distance)

    if verbose:
        err([closest, indices_A, unmatched_i_B, cost])
    return closest, indices_A, unmatched_i_B, cost


def process_reordering_costs(A, B, closest, indices_B, cost=0.0):
    """
    Calculate the reordering cost.  Handle some overhead
    """
    verbose = False
    if verbose:
        err([A, B, closest, indices_B, cost])
        print('\nAttempting to reorder:\n "%s" "%s"'% (' '.join(A), ' '.join(B)))
        print('\nClosest: (not in order)')
        for k,v in closest.items():
            kA = A[k]
            vB = B[v]
            print('\t', kA, ' : ', vB)

    out_A = []
    out_B = []
    last_i_B = -1

    # Iterate over all tokens in A.
    #     Add to the reordering cost whenever B indices come out of order
    for i_A,token_A in enumerate(A):

        skip_cost = marginal_cost(i_A, token_A, None, None)
        if verbose:
            err([token_A, "skip_cost:", skip_cost])
        
        # If next i_B unmatched, incur skip cost
        i_B = last_i_B + 1
        while i_B in indices_B:
            cost += skip_cost              # Cost of skipping
            if verbose:
                err(["cost:", cost])
            token_B = B[i_B]
            out_A.append(None)             # Add to out_ arrays
            out_B.append(token_B)
            i_B += 1
        
        if i_A in closest:                 # For token_A there is a token_B
            i_B = closest[i_A]
            token_B = B[i_B]
            out_A.append(token_A)          # Add to out_ arrays
            out_B.append(token_B)

            # Calculate reorder cost for this token
            reorder_cost = marginal_cost(i_A, token_A, i_B, token_B)
            if i_B < last_i_B:
                cost += reorder_cost       # Reordering cost only if out of order
                if verbose:
                    err(["cost:", cost])
                
        else:                              # There is no token_B for this token_A
            cost += skip_cost              # Cost of skipping
            if verbose:
                err(["Skipping for:", token_A])
            out_A.append(token_A)          # Add to out_ arrays
            out_B.append(None)
            
        last_i_B = i_B

    if verbose:
        err([out_A, out_B])
        print('\nReordered    A\t  :  B')
        for i in range(len(out_A)):
            print(i, '\t', out_A[i], '\t : ', out_B[i])
        print ('cost:', cost)

    return cost


def holistic_cost(A, B, closest, indices_A, indices_B, cost, length, verbose=False):
    """
    Compute a holistic sentence-level fuzzy weighted edit-distance score
    """
    if verbose:
        err([A, indices_A, B, indices_B, closest, cost, length])
        print('\nAttempting to reorder:\n\t"%s"\n\t"%s"'% (' '.join(A), ' '.join(B)))
        print('\nClosest: (not in order?)')
        for k,v in closest.items():
            token_A = A[k]
            token_B = B[v]
            print('\t', token_A, ' : ', token_B)

    # Add reordering costs for matches
    for k,v in closest.items():
        token_A = A[k]
        token_B = B[v]
        new_cost = (abs(k-v)/length)
        cost += new_cost
        if verbose:  print('\t', token_A, ' : ', token_B, '  cost:', new_cost)

    # Add costs for unmatched tokens and normalize
    num_matched    = 2 * len(closest)
    num_unmatched  = len(indices_A) + len(indices_B)
    unmatched_cost = num_unmatched / (num_unmatched + num_matched)
    untapped_cost  = 1 - cost
    rel_um_cost    = unmatched_cost * untapped_cost
    if verbose:
        err([num_unmatched, num_matched, rel_um_cost, cost])
    cost          += rel_um_cost
    cost           = min(1, cost)

    return cost

    
def cost_best_reorder(A, B, length, verbose=True):
    """
    Reorders the tokens in B to best match those in A.

    Greedily matches from longest to shortest tokens in B
    """
    # Perfect matches first  (no cost associated with these matches)
    closest, indices_A, indices_B = process_perfect_matches(A, B)

    # Then best matches  (cost calculations begin here)
    closest, indices_A, indices_B, cost = process_best_matches(A, B, closest, indices_A, indices_B)
    
    # Process reordering costs
    # rcost = process_reordering_costs(A, B, closest, indices_B, cost)
    hcost = holistic_cost(A, B, closest, indices_A, indices_B, cost, length, verbose)

    return hcost


def meaningful_length(A):
    """
    Number of meaningful words in a phrase
    """
    m = 0
    for a in A:
        if a in low_val_words:
            pass
        elif a in medium_val_words:
            m += 0.5
        else:
            m += 1
    return m


def phrase_similarity(a, b, verbose=False):
    """
    Similarity score between two strings, a and b

    Modified bi-scale (words and characters) Levenshtein-Damerau
    
    Parameters
    ----------
    a : str
    b : str

    Returns
    -------
    score : float
        between [0,1]
    """
    a = simplify_for_distance(a)
    b = simplify_for_distance(b)
    if a == b:
        return 1.0   # Perfect Match
    
    A = a.split(' ')
    B = b.split(' ')
    A1 = B1 = word_trans = None
    length = cost = 0.0

    if len(A) >= len(B):
        length = meaningful_length(A)
        cost = cost_best_reorder(A, B, length, verbose)   # Reorder so the closest words line up
        if verbose:
            err([A, B, length, cost, 1-cost])
    else:
        length = meaningful_length(B)
        cost = cost_best_reorder(B, A, length, verbose)   # Reorder so the closest words line up
        if verbose:
            err([A, B, length, cost, 1-cost])

    score = 1 - cost

    if verbose:
        err([a, b, length, cost, score])
            
    return score
    

################################################################################
# MAIN

if __name__ == '__main__':

    # Compute Levenshtein distance
    if sys.argv[1] == '-l':
        l = levenshtein(sys.argv[2], sys.argv[3])
        r = levenshtein_r(sys.argv[2], sys.argv[3])

        print ('\nIterative:',l)
        print ('Recursive:',r)

    # Compute Damerau-Levenshtein distance
    if sys.argv[1] == '-d':
        l = damerauLevenshtein(sys.argv[2], sys.argv[3], cost=letterCost_suppress_vowels)
        # r = damerauLevenshtein_r(sys.argv[2], sys.argv[3])

        print ('\n\nCost:',l)
        # print ('Recursive:',r)

    # Compare phrases
    elif sys.argv[1] == '-p':
        l = damerauLevenshtein_phrase(sys.argv[2], sys.argv[3])
        r = damerauLevenshtein_phrase_r(sys.argv[2], sys.argv[3])

        print ('\nIterative:',l)
        print ('Recursive:',r)

    # Phase Similarity
    elif sys.argv[1] == '--ps':
        score = phrase_similarity(sys.argv[2], sys.argv[3])
        print ('Score:', score)
        
    # String Distance
    elif sys.argv[1] == '--sd':
        dist = string_distance(sys.argv[2], sys.argv[3])
        print ('Dist:', dist)
        

################################################################################
################################################################################
