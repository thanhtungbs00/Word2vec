from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, string
import os, sys, re
import collections, argparse

import pandas as pd
import numpy as np
from pattern.en import tag

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

from contractions import contractions_dict



'''--------- Global variable -----------------'''
import pprint
pprint = pprint.pprint
DATASETS = "datasets/"

stopword_list = nltk.corpus.stopwords.words('english')
stopword_list = stopword_list + ['mr', 'mrs', 'come', 'go', 'get', 'tell', 
                                 'listen', 'one', 'two', 'three', 'four', 
                                 'five', 'six', 'seven', 'eight', 'nine', 
                                 'zero', 'join', 'find', 'make', 'say', 
                                 'ask', 'tell', 'see', 'try', 'back', 'also']
wnl = WordNetLemmatizer()
# create an object of stemming function
stemmer = SnowballStemmer("english")


def load_Data(filename, path=DATASETS):
    data = []
    txt_path = os.path.join(path, filename)
    file = open(txt_path, "r").readlines()
    for line in file:
        data.append(line)
    return data

''' Preprocessing '''

def stemming(text):    
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

def remove_newline(text):
    return text.rstrip()

def remove_punctuation(text):
    '''a function for removing punctuation'''
    import string
    # punctuation : !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    # replacing the punctuations with no space, 
    # which in effect deletes the punctuation marks 
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    return text.translate(translator)

def convert_lower(text):
    return text.lower()

'''--------- Expanding contraction -----------'''
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'","", expanded_text)
    return expanded_text

# Annotate text tokens with POS tags
def pos_tag_text(text):
    # convert Penn treebank tag to wordnet tag
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    tagged_text = tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag)) for word, pos_tag in tagged_text]
    return tagged_lower_text

# lemmatize text based on POS tags
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

# special characters
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# stop words
def remove_stopwords(text):
    tokens = tokenize_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# remove special character
def keep_text_characters(text):
    '''a function which remove special characters like "?, ~, !, .." '''
    filtered_tokens = []
    tokens = tokenize_text(text)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

import html
# final function text normalization
def normalize_corpus(corpus, lemmatize=True, only_text_chars=False, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        # Convert all named and numeric character references (e.g. &gt;, &#x3e;)to the corresponding Unicode characters
        text = html.unescape(text) 
        text = expand_contractions(text, contractions_dict)
        if lemmatize:
            text = lemmatize_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if only_text_chars:
            text = keep_text_characters(text)
        if tokenize:
            text = tokenize_text(text)
        normalized_corpus.append(text)
        
    return normalized_corpus
#--------------------------------

def smooth_data(data):
    data['text'] = data['text'].apply(convert_lower)
    data['text'] = data['text'].apply(remove_newline)
    #data['text'] = data['text'].apply(remove_punctuation)        
    #data['text'] = data['text'].apply(stemming)
    #data['text'] = data['text'].apply(keep_text_characters)
    
    return data

''' Word2vec '''

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def main():
    # corpos = load_Data("Shipper.txt")
    corpos = load_Data("POR.txt")
    corpos = pd.DataFrame(corpos, columns=['text'])
    
    ''' create a test set '''
    train_set, test_set = split_train_test(corpos, 0.9)
    print("Size of training dataset:", len(train_set))
    print("Size of testing dataset:", len(test_set))

    print(train_set.info())

    '''
    #  --- pre-processing--- 
    print(" Before pre-processing: ")
    pprint(train_set)
    
    
    #smooth_data(training_data)
    
    print(" After pre-processing: ")
    pprint(train_set)
    '''
if __name__ == "__main__":
    main()
    