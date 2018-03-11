# -*- coding: utf-8 -*-
# encoding=utf8  
# The above lines enables the usage of non-ascii chars in this code
""" normalize.py

Normalize raw text:  Remove strange encodings and strange strings of text.

"""
import sys  
#reload(sys)  
#sys.setdefaultencoding('utf8')

import os, re
import chardet
import datetime
import unicodedata

from gmutils.utils import err, argparser
from gmutils.objects import Options

################################################################################

def alpha_only(line):
    """
    Return a string having only alphabet characters for dictionary checking

    """
    out = ''
    for l in line:
        if re.search(u'[a-zA-ZñÑ \'\-]', l):
            out += l
            
    return out


def final_clean(text):
    """
    Remove unneeded punctuation and spaces

    """
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.+', '.', text)        # Remove extra periods
    text = re.sub(r'\-+\s*$', '', text)     # Remove ending dashes
    text = re.sub(r"^'+", '', text)
    
    text = re.sub(r' +', ' ', text)         # Remove unneeded spaces
    text = re.sub(r'^\s+', '', text)
    text = re.sub(r'\s+$', '', text)

    return text


def no_urls(text):
    """
    Split tokens, then remove all that look someone similar to URLs or Emails

    """
    tokens = []
    for token in text.split():
        if re.search("\@", token)  or  re.search("/", token):
            pass
        else:
            tokens.append(token)

    final = " ".join(tokens)
    return final


def no_strange_tokens(text):
    """
    Split tokens, then remove all tokens the mix any of the following categories:

      - letters
      - numbers
      - rare special chars (for instqnce, chars not from the common set: .,'"?!-)

    """
    tokens = []
    lastWasAcronym = False
    for token in text.split():
        tokenL = token.lower()
        
        # if re.search("[a-z].*\d", tokenL)  or  re.search("\d.*[a-z]", tokenL):   # Remove num/letter mix
        #    pass
        
        if re.search(u"[a-z]\'s$", tokenL):               # 's contraction, possessives
            tokens.append(token)            
        elif re.search(u"[a-z]\'re$", tokenL):            # 're contraction
            tokens.append(token)            
        elif re.search("[a-z]\'d$", tokenL):              # 'd contraction
            tokens.append(token)            
        elif re.search("[a-z]n\'t$", tokenL):             # 't contraction
            tokens.append(token)
        elif re.search("[a-z]n\'ll$", tokenL):            # 'll contraction
            tokens.append(token)
            
        elif re.search("\'", token)  and  re.search("[a-z]", tokenL) :      # Look at caps version, Remove if random ' found
            pass

        # Remove some lonely and unneeded punctuation
        elif re.search("\?\?+", token):
            pass

        # DEFAULT behavior
        else:
            tokens.append(token)                          # Accept by default

    final = " ".join(tokens)
    return final


def scrub_charByChar(text):
    """
    Char-by-char scrubbing.  Takes str, Returns str

    """
    verbose = False
    try:
        final = ""

        text = re.sub('\s', ' ', text)   # Standardize on whitespace
        
        for t in text:
            if re.search(u'[ 0-9a-zA-ZñÑ\.,\'\?\!\"\:\&\$\%\@\|\_]', t):
                final += t        # keep the char
                if verbose:
                    arr = [final]
                    err([arr, [t]])
                    
            elif re.search(u'[—\-]', t):
                final += '-'      # Replace with normalized dash
                if verbose:
                    arr = [final]
                    err([arr, [t]])
                    
            elif re.search(u'[\;]', t):
                final += t        # Replace with a period
                if verbose:
                    arr = [final]
                    err([arr, [t]])
                    
            elif re.search(u'\n', t):
                final += '\n'     # Keep as newline
                if verbose:
                    arr = [final]
                    err([arr, [t]])
                    
            else:   # Drop the char
                if verbose:
                    arr = [final]
                    err([arr, [t]])
                    
        text = final
        if verbose:
            arr = [text]
            err([arr])
                    
        return text
    except Exception as e:
        err([], {'exception':e, 'exit':True})


def ascii_fold(s):
   return ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))


def normalize(text, options=None):
    """
    Normalize a string of text, assuming standard English language

    Parameters
    ----------
    text : str

    Options
    -------
    verbose : boolean

    no_urls : boolean

    Returns
    -------
    str

    """
    options = Options(options)
    # options.set('verbose', True)

    if options.get('verbose'):
        err([[text]])

    if options.get('remove_citations'):
        text = remove_citations(text)
        
    if options.get('verbose'):
        err([[text]])

    # Punctuation normalization
    text = re.sub(r"“\s+", '“', text)
    text = re.sub(r"\s+”", '”', text)
    text = re.sub(r"—", '-', text)
    text = re.sub(r"–", '-', text)
    text = re.sub(r"\xe2\x80\x99", "'", text)
    text = re.sub(r"''+", '"', text)
    text = re.sub(r"``+", '"', text)
    text = re.sub(r"“+", '"', text)
    text = re.sub(r"”+", '"', text)
    text = re.sub(r"‘", "'", text)
    text = re.sub(r"’", "'", text)
    text = re.sub(r"`", "'", text)
    text = re.sub(r"—", "-", text)
    text = re.sub(r"\u2013", "-", text)
    text = re.sub(r"\-+", "-", text)
    text = re.sub(r"\.*…+\.*", "...", text)

    if options.get('verbose'):
        err([[text]])

    # Char-by-char scrubbing
    text = scrub_charByChar(text)
    if options.get('verbose'):
        err([[text]])

    # Some final options
    if options.get('no_urls'):                 # Remove URLs and Emails if requested in options
        text = no_urls(text)

    text = final_clean(text)
    if options.get('verbose'):
        err([[text]])

    return text


def clean_spaces(line):
    """
    Deal with extra spaces

    """
    line = re.sub('\s+', ' ', line)
    line = re.sub('^\s+', '', line)
    line = re.sub('\s+$', '', line)
    return line


def remove_citations(line):
    """
    Removes things that look like citations from Wikipedia-sourced data
    """
    # line = re.sub(r"\[citation needed\]", '', line)
    line = re.sub(r"(?<=[a-zA-Z])\.\d[\d,]*$", '.', line)
    line = re.sub(r"(?<=[a-zA-Z])\.\d[\d,]* \s*([A-Z][a-z])", r'.  \1', line)
    line = re.sub(r"(?<=[a-zA-Z])\.\d[\d,]* \s*([A-Z] )", r'.  \1', line)
    line = re.sub(r"\<[\w\.\,\'\-\?\!\%\&\*\@ ]+\>", "", line)
    
    return line


def ends_with_punctuation(line):
    if re.search(r'[\.\,\;\:\'\"\?\!]$', line):
        return True
    return False

    
def simplify(text):
    """
    Return a string having only alphabet characters and basic punctuation

    """
    verbose = False
    try:
        output = ''
        text = text.rstrip()
        if verbose:
            err([[text]])
            
        text = normalize(text)
        if verbose:
            err([[text]])

        # Iterate over characters
        for x in text:
            # Previously filtered char-by-char: (u'[0-9a-zA-ZñÑ\.\'\?\!\"\:\$\%\@]', t)
            # char-by-char scrubbing
            if re.search(u'[\dA-ZñÑ\'\-\s\?\!\.\,\;]', x, flags=re.I):   # further simplification / sanity check
                output += x

        if verbose:
            err([[output]])
            
        # Some clean-up
        output = clean_spaces(output)
        if verbose:
            err([[output]])
        
        output = re.sub('^and\s+', '', output)                       # Remove initial "and"
        if verbose:
            err([[output]])
            
        return(output)
    except Exception as e:
        err([], {'exception':e, 'exit':True})


def simplify_for_distance(line):
    """
    Simplify a string for the express purpose of computing an optimistic Levenshtein-Damerau distance

      - lower case
      - normalized
      - ASCII-folded
      - remove extra space

    """
    line = line.lower()
    line = simplify(line)
    line = ascii_fold(line)
    line = re.sub('-', '_', line)
    out = ''
    for l in line:
        if re.search(u'[0-9a-zA-ZñÑ _]', l):
            out += l
    line = clean_spaces(out)
    
    return line


def close_enough(A, B):
    """
    Determine if two words are similar "enough" (useful for many situations)

    Parameters
    ----------
    A : str

    B : str

    Returns
    -------
    boolean

    """
    if A == B:
        return True
    
    a = simplify_for_distance(A)
    b = simplify_for_distance(B)
    if re.search(r'[a-zA-Z0-9]', a)  and  re.search(r'[a-zA-Z0-9]', b):
        pass
    else:
        return False
    
    if a == b:
        return True
    
    return False

    
################################################################################
###  MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "normalize.py"})
    parser.add_argument('--ascii_fold', help='Fold special characters to ASCII', required=False, type=str)
    args = parser.parse_args()   # Get inputs and options

    if args.str:
        print(normalize(args.str[0], options={'verbose':args.verbose, 'remove_citations':True}))

    elif args.ascii_fold:
        print(ascii_fold(args.ascii_fold))


################################################################################
################################################################################
