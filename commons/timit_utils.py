# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:37:32 2019

@author: isaac
"""
import numpy as np
from itertools import groupby
from numbers import Number

CATEGORIES_OPTIONS = {"scanlon", "reynolds", "standard"}

phons_all_keys = ["b", "bcl", "d", "dcl", "g", "gcl", "p", "pcl", "t", "tcl", "k", "kcl", "dx", "q", "jh", "ch", "s", "sh", "z", "zh", "f",
        "th", "v", "dh", "m", "n", "ng", "em", "en", "eng", "nx", "l", "r", "w", "y", "hh", "hv", "el", "iy", "ih", "eh", "ey",
        "ae", "aa", "aw", "ay", "ah", "ao", "oy", "ow", "uh", "uw", "ux", "er", "ax", "ix", "axr", "ax-h", "pau", "epi", "h#"]

phons_redu_keys = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'h', 'ih', 'iy', 'jh', 'k', 'l', 'm',
             'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z','-']

redu_dict = {"aa":"aa", "ao":"aa", "ah":"ah", "ax":"ah", "ax-h":"ah", "er":"er", "axr": "er", "hh":"h", "hv":"h", "jh":"jh", "d":"d",
             "k":"k", "s":"s", "g":"g", "r":"r", "w":"w", "dx":"dx", "y":"y", "uh":"uh", "ae":"ae", "oy":"oy", "dh":"dh", "iy":"iy",
             "v":"v", "f":"f", "t":"t", "ow":"ow", "ch":"ch", "b":"b", "ay":"ay", "th":"th", "ey":"ey", "p":"p", "aw":"aw", "z":"z",
             "eh":"eh", "ih":"ih","ix":"ih", "el":"l", "l":"l", "em":"m", "m":"m", "en":"n", "n":"n", "nx":"n", "eng":"ng","ng":"ng",
             "zh":"sh", "sh":"sh", "ux":"uw", "uw":"uw", "pcl":"sil", "tcl":"sil", "kcl":"sil", "bcl":"sil", "dcl":"sil", "gcl":"sil",
             "h#":"sil", "pau":"sil", "epi":"sil", "q":"-"} #no se si debería quitar los h#                 
             
             
#Reynolds, T.J. & Antoniou, C.A. (2003). Experiments in speech recognition using a modular
#MLP architecture for acoustic modelling. Information Sciences, vol 156, Issue 1-2,
#2003, pp 39 – 54, ISSN 0020-0255. 
cat_reynolds_dict = {'t': 'plosives', 'g': 'plosives', 'k': 'plosives','jh': 'plosives',
                'b': 'plosives', 'd': 'plosives', 'p': 'plosives', 'ch': 'plosives',
                'h': 'fricatives', 'sh': 'fricatives', 'z': 'fricatives', 'f': 'fricatives',
                'v': 'fricatives', 'dh': 'fricatives', 's': 'fricatives', 'th': 'fricatives',
                'm': 'nasals', 'n': 'nasals', 'ng': 'nasals', 
                'r': 'semivowels', 'w': 'semivowels', 'er': 'semivowels', 'l': 'semivowels', 'y': 'semivowels',
                'uh': 'vowels', 'ae': 'vowels', 'ih': 'vowels', 'iy': 'vowels',
                'eh': 'vowels', 'ah': 'vowels', 'aa': 'vowels', 'uw': 'vowels',
                'aw': 'diphthongs', 'ay': 'diphthongs', 'oy': 'diphthongs', 'ow': 'diphthongs', 'ey': 'diphthongs',
                'dx': 'closures', 'sil': 'closures',
                '-': 'none'}

cat_reynolds_keys = ['plosives', 'fricatives', 'nasals', 'semivowels', 'vowels', 'diphthongs', 'closures']


#Scanlon, P.; Ellis, D. & Reilly, R. (2007). Using Broad Phonetic Group Experts for Improved
#Speech Recognition. IEEE Transactions on Audio, Speech and Language Processing,
#vol.15 (3) , pp 803-812, March 2007, ISSN 1558-7916. 
#cat_scanlon_dict = {'aw': 'vowels', 'ay': 'vowels', 'uh': 'vowels', 'r': 'vowels', 'w': 'vowels', 'oy': 'vowels', 
#                    'er': 'vowels', 'ae': 'vowels', 'ih': 'vowels', 'iy': 'vowels', 'l': 'vowels', 'ow': 'vowels',
#                    'ah': 'vowels', 'eh': 'vowels', 'y': 'vowels', 'aa': 'vowels', 'uw': 'vowels', 'ey': 'vowels',
#                    't': 'stops', 'g': 'stops', 'k': 'stops','jh': 'stops',
#                    'b': 'stops', 'd': 'stops', 'p': 'stops', 'ch': 'stops',
#                    'h': 'fricatives', 'sh': 'fricatives', 'z': 'fricatives', 'f': 'fricatives',
#                    'v': 'fricatives', 'dh': 'fricatives', 's': 'fricatives', 'th': 'fricatives',
#                    'm': 'nasals', 'n': 'nasals', 'ng': 'nasals',
#                    'dx': 'silences', 'sil': 'silences',
#                    '-': 'none'}
#
#cat_scanlon_keys = ['vowels', 'stops', 'fricatives', 'nasals', 'silences']


cat_scanlon_dict = {'aw': 'vocales', 'ay': 'vocales', 'uh': 'vocales', 'r': 'vocales', 'w': 'vocales', 'oy': 'vocales', 
                    'er': 'vocales', 'ae': 'vocales', 'ih': 'vocales', 'iy': 'vocales', 'l': 'vocales', 'ow': 'vocales',
                    'ah': 'vocales', 'eh': 'vocales', 'y': 'vocales', 'aa': 'vocales', 'uw': 'vocales', 'ey': 'vocales',
                    't': 'stops', 'g': 'stops', 'k': 'stops','jh': 'stops',
                    'b': 'stops', 'd': 'stops', 'p': 'stops', 'ch': 'stops',
                    'h': 'fricativos', 'sh': 'fricativos', 'z': 'fricativos', 'f': 'fricativos',
                    'v': 'fricativos', 'dh': 'fricativos', 's': 'fricativos', 'th': 'fricativos',
                    'm': 'nasales', 'n': 'nasales', 'ng': 'nasales',
                    'dx': 'silencios', 'sil': 'silencios',
                    '-': 'none'}

cat_scanlon_keys = ['vocales', 'stops', 'fricativos', 'nasales', 'silencios']



cat_standard_dict = {"aa":"vowel","ae":"vowel","ah":"vowel","aw":"vowel","ay":"vowel",
                     "eh":"vowel","er":"vowel","ey":"vowel",
                     "iy":"vowel","ih":"vowel",
                     "ow":"vowel","oy":"vowel","uh":"vowel","uw":"vowel",
                     "ch":"affricative", "jh":"affricative",
                     "dh":"fricative", "f":"fricative", "s":"fricative", "sh":"fricative", "th":"fricative", "v":"fricative", "z":"fricative",             
                     "b":"stop", "d":"stop", "dx":"stop", "g":"stop", "k":"stop", "p":"stop", "t":"stop",
                     "sil":"stop",
                     "m":"nasal", "n":"nasal", "ng":"nasal",                     
                     "l":"semivowel", "h":"semivowel", "r":"semivowel", "w":"semivowel", "y":"semivowel",  
                     "-":"none"}

cat_standard_keys = ['affricative', 'fricative', 'nasal', 'none', 'semivowel', 'stop', 'vowel'],

#Neural speech recognition: Continuous phoneme decoding using 
#spatiotemporal representations of human cortical activity
cat_moses_dict = {"silence": {"sil"},
                    "stop":	{"b", "d", "g", "p", "t", "k"},
                    "affricative": {"ch", "jh"},
                    "fricative": {"f", "v", "s", "z", "sh", "th", "dh", "h"},
                    "nasal": {"m", "n", "ng"},
                    "approximant": {"w", "y", "l", "r"},
                    "monophthong": {"iy", "aa", "ae", "eh", "ah", "uw", "ao", "ih", "uh", "er"},
                    "diphthong": {"ey", "ay", "ow", "aw", "oy"}}
cat_moses_keys = ['silence', 'stop', 'affricative', 'fricative', 'nasal', 'approximant', 'monophthong', 'diphthong']

def list_inception(mylist):
    if isinstance(mylist, np.ndarray):
        mylist = mylist.tolist()
        
        
    if isinstance(mylist, Number) or isinstance(mylist, str):
        return 0 #int or str
    elif isinstance(mylist[0], Number) or isinstance(mylist[0], str):
        return 1 #list of int or str
    elif isinstance(mylist[0][0], Number) or isinstance(mylist[0][0], str):
        return 2 #list of lists
    raise Exception("too deep")
    return -1 #too deep

def remove_q(y):
    """Remove q from list or list of lists of reduced index phons. Example y = [32,1 25, 3,...]"""
    def _remove_q_list(mlist):
        return np.array([x for x in mlist if x != index_to_remove])

    y = np.array(y)

    deepness = list_inception(y)
    index_to_remove = phons_redu_keys.index("-")
    if deepness == 0:
        raise Exception("You should provide something deeper, a list at least")
    elif deepness == 1:
        return _remove_q_list(y)
    else:
        return np.array([_remove_q_list(sublist) for sublist in y])

def to_ctc(y):
    def _to_ctc_list(mlist):
        return np.array([k for k,g in groupby(mlist)])
    
    y = np.array(y)
    
    deepness = list_inception(y)
    if deepness == 0:
        raise Exception("You hould provide something deeper, a list at least")
    elif deepness == 1:
        return _to_ctc_list(y)
    elif deepness == 2:
        return np.array([_to_ctc_list(sublist) for sublist in y])

def labels_to_int(y, reduced = True):
    def _labels_to_int(labels):
        if reduced:
            return np.array([phons_redu_keys.index(y) for y in labels])
        else:
            return np.array([phons_all_keys.index(y) for y in labels])
        
    y = np.array(y)

    deepness = list_inception(y)
    if deepness == 0:
        return _labels_to_int([y])
    elif deepness == 1:
        return _labels_to_int(y)
    else:
        return np.array([_labels_to_int(sublist) for sublist in y])

def labels_to_str(y, reduced = True):
    def _labels_to_str(labels):
        if reduced:
            return np.array([phons_redu_keys[int(y)] for y in labels])
        else:
            return np.array([phons_all_keys[int(y)] for y in labels])
            
    y = np.array(y)

    deepness = list_inception(y)
    if deepness == 0:
        return _labels_to_str([y])[0]
    elif deepness == 1:
        return _labels_to_str(y)
    else:
        return np.array([_labels_to_str(sublist) for sublist in y])
    
def reduce_labels(y, input_index = True, output_index = True):
    def _reduce_labels(labels):
        if output_index: #return ints
            return np.array([phons_redu_keys.index(redu_dict[y]) for y in labels])
        else: #return strings
            return np.array([redu_dict[y] for y in labels])
        
    if input_index:
        y = labels_to_str(y, False)

    deepness = list_inception(y)
    if deepness == 0:
        return _reduce_labels([y])[0]
    elif deepness == 1:
        return _reduce_labels(y)
    else:
        return np.array([_reduce_labels(sublist) for sublist in y])
    
def labels_to_category(y, category_type = "scanlon", input_index = True, output_index = True):
    def _labels_to_category(labels):
        if category_type == "scanlon":
            if output_index:
                return [cat_scanlon_keys.index(cat_scanlon_dict[y]) for y in labels]
            else:
                return [cat_scanlon_dict[y] for y in labels]
        elif category_type == "reynolds":
            if output_index:
                return [cat_reynolds_keys.index(cat_reynolds_dict[y]) for y in labels]
            else:
                return [cat_reynolds_dict[y] for y in labels]
        elif category_type == "standard":
            if output_index:
                return [cat_standard_keys.index(cat_standard_dict[y]) for y in labels]
            else:
                return [cat_standard_dict[y] for y in labels]
        raise Exception("Wrong category name")
            
    if input_index:
        y = labels_to_str(y)
        
    deepness = list_inception(y)
    if deepness == 0:
        return _labels_to_category([y])[0]
    elif deepness == 1:
        return _labels_to_category(y)
    else:
        return np.array([_labels_to_category(sublist) for sublist in y])
    
def categories_to_int(categories, category_type = "scanlon"):
    def _categories_to_int(labels):
        if category_type == "scanlon":
            return [cat_scanlon_keys.index(y) for y in labels]
        elif category_type == "reynolds":
            return [cat_reynolds_keys.index(y) for y in labels]
        elif category_type == "standard":
            return [cat_standard_keys.index(y) for y in labels]
        raise Exception("Wrong category name")
                    
    deepness = list_inception(categories)
    if deepness == 0:
        return _categories_to_int([categories])[0]
    elif deepness == 1:
        return _categories_to_int(categories)
    else:
        return np.array([_categories_to_int(sublist) for sublist in categories])
#    return [_categories_to_int.index(c) for c in categories]
