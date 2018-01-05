# -*- coding: utf-8 -*-
# encoding=utf8  
# The above lines enables the usage of non-ascii chars in this code
""" normalize.py

Normalize raw text:  Remove strange encodings and strange strings of text.

"""
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

import os, re
import chardet
import HTMLParser
import cPickle as pickle
import datetime

# Handling of special encodings
# If pip doesn't install it correctly:
#    Download Wheel file: https://pypi.python.org/pypi/Unidecode#downloads
#    pip install Unidecode-0.04.20-py2.py3-none-any.whl
import unidecode

html_parser = HTMLParser.HTMLParser()  # used in normalizing text, removes some strange characters

from dautils import err, argparser
from daobjects import Options


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
            if re.search(u'[ 0-9a-zA-ZñÑ\.\'\?\!\"\:\$\%\@\|\_]', t):
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


def ascii_fold(text):
    """
    Char-by-char folding

    """
    b = None
    if isinstance(text, unicode):
        b = text.encode('latin1')
    elif isinstance(text, str):
        b = text
    else:
        err()
        exit()

    b = re.sub(r'\xc3\x91', 'N', b)
    b = re.sub(r'\xd1', 'N', b)
        
    try:
        c = b.decode('utf8')
        decoded = unidecode.unidecode(c)
        return decoded
    except:
        decoded = scrub_charByChar(text)
        return decoded
        

def normalize(text, options=None):
    """
    Normalize a string of text, assuming standard English language

    Parameters
    ----------
    text : str

    Options
    -------
    NOTE: 'options' can be passed either as a dict or an Options object, e.g. both of these are acceptable:
        options = {'verbose':True}
        options.verbose = True      (See utils.py for Options class definition)

    verbose : boolean

    remove_mixed : boolean
        Set to True if you want to remove mixtures of numbers and letters, e.g. "EUR410bn"

    no_consecutive_capitals : boolean
        Removes tokens having two or more consecutive capitals

    remove_parentheticals : boolean
        Removes parenthetical statements

    no_urls : boolean
        Removes tokens that look like URLs

    no_strange_tokens : boolean
        Designed to handle some particularly junk-filled files

    show_warnings : boolean
        Will warn for out-of-language words

    Returns
    -------
    str

    """
    options = Options(options)
    # options.set('verbose', True)
        
    if options.get('verbose'):
        tarray = [text, u'N1 Ñ', u'N2 \xd1', u'N3 \xc3\x91']
        err([tarray])

    try:
        ##########################################################################################################################
        # Decoding
    
        if text == '':    # Chardet can't handle empty strings
            return ''

        ##  Re: N with tilde
        #   seen in input:           \xc3\x91
        #   seen in decoded output:  \xd1

        
        ##  Handle enyas first  ####################
        if options.get('verbose'):
            err([[text], text])
            
        try:
            text = re.sub(u'\xc3\x91', u'\xd1', text, flags=re.UNICODE)
        except Exception as e:
            pass
        if options.get('verbose'):
            err([[text], text])
                                        
        try:
            text = re.sub(r'\xc3\x91', u'\xd1', text)
        except Exception as e:
            err([[text]], {'exception':e})
        if options.get('verbose'):
            err([[text], text])
            
        ##  Enyas handled  ####################

        ########################
        ##   DECODING CHAIN   ##
        ########################

        # First make the str 'b'
        if isinstance(text, unicode):
            text = text.encode('latin1')
        elif isinstance(text, str):
            text = text
        else:
            err()
            exit()

        # Then attempt to deocde to 'utf8'
        try:
            text = text.decode('utf8')
        except:

            ###  Longer decoding chain  ###
            while not isinstance(text, unicode):  # if unicode, Leave alone.  We want unicode

                encoding = None
                try:
                    encoding = chardet.detect(text)['encoding']
                    if encoding is not None:
                        text = text.decode(encoding)             # Unencode where possible using sensed encoding
                        if options.get('verbose'):
                            err([[text], text])
                except Exception as e:
                    err([encoding], {'exception':e})

                if isinstance(text, unicode):  # if unicode, Leave alone.  We want unicode
                    break

                ############################################
                try:
                    text = unidecode.unidecode(text)                 # Decode other non-ASCII
                    if options.get('verbose'):
                        err([[text]])
                except Exception as e:
                    err([], {'exception':e})

                if isinstance(text, unicode):  # if unicode, Leave alone.  We want unicode
                    break

                ################################################
                try:
                    text = text.decode('utf-8')          # Unencode where possible using utf-8
                    if options.get('verbose'):
                        err([[text]])
                except Exception as e:
                    err([], {'exception':e})

                if isinstance(text, unicode):  # if unicode, Leave alone.  We want unicode
                    break

                ##############################################
                try:
                    text = text.decode('ascii')      # Unencode where possible using ascii
                    if options.get('verbose'):
                        err([[text]])
                except Exception as e:
                    err([], {'exception':e})
                    return ''                        # at this point, give up

                ##############################################

        if text == '':                               # Unidecode can't handle empty strings either, but chardet might generate them
            return ''                                # In this case, just return
    
        if options.get('verbose'):
            err([[text]])

        # Some punctuation normalization
        text = re.sub(r"“\s+", '“', text)
        text = re.sub(r"\s+”", '”', text)
        text = re.sub(r"—", '-', text)

        if options.get('verbose'):
            tarray = [text]
            err([tarray])
        
        text = html_parser.unescape(text)                # Remove HTML encodings

        if options.get('verbose'):
            tarray = [text]
            err([tarray])
        
        if options.get('remove_parentheticals'):
            text = re.sub(r'\s*\(.*\)\s*', ' ', text)    # Remove parenthetical expressions

        if options.get('verbose'):
            tarray = [text]
            err([tarray])
        
        ##########################################################################################################################
        # Further punctuation normalization
    
        text = re.sub(r"\xe2\x80\x99", "'", text)
        text = re.sub(r"''+", '"', text)
        text = re.sub(r"``+", '"', text)
        text = re.sub(r"“+", '"', text)
        text = re.sub(r"”+", '"', text)
        text = re.sub(r"‘", "'", text)
        text = re.sub(r"’", "'", text)
        text = re.sub(r"`", "'", text)
        text = re.sub(r"—", "-", text)
        text = re.sub(r"\-+", "-", text)
        text = re.sub(r"\.*…+\.*", "...", text)

        if options.get('verbose'):
            tarray = [text]
            err([tarray])
        
        # Char-by-char scrubbing
        text = scrub_charByChar(text)
    
        if options.get('verbose'):
            tarray = [text]
            err([tarray])

        # Some final options
        if options.get('no_urls'):                 # Remove URLs and Emails if requested in options
            text = no_urls(text)
        
        if options.get('no_strange_tokens'):       # Designed to handle some particularly junk-filled files
            text = no_strange_tokens(text)
        
        text = final_clean(text)
    
        if options.get('verbose'):
            tarray = [text]
            err([tarray])
        
        return text
    
    except Exception as e:
        err([], {'exception':e})


def clean_spaces(line):
    """
    Deal with extra spaces

    """
    line = re.sub('\s+', ' ', line)
    line = re.sub('^\s+', '', line)
    line = re.sub('\s+$', '', line)
    return line
    
    
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
            
        text = normalize(text) #, {'verbose':True})                 # Module for text simplification
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
    """
    line = ascii_fold(line)
    line = re.sub('-', '_', line)
    out = ''
    for l in line:
        if re.search(u'[0-9a-zA-ZñÑ _]', l):
            out += l
    line = clean_spaces(out)
    return line


################################################################################
###  MAIN

if __name__ == '__main__':

    parser = argparser({'desc': "normalize.py"})
    args = parser.parse_args()   # Get inputs and options

    if args.text:
        print(normalize(args.text[0]))


################################################################################
################################################################################
