# -*- coding: utf-8 -*-
"""
Created on Wed Jul 01 23:48:24 2015

@author: ngaude
"""

# import and setup modules we'll be using in this notebook
import time
import logging
import itertools

import numpy as np
import gensim

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO  # ipython sometimes messes up the logging setup; restore

corpus = 'E:/workspace/data/simplewiki-20140623-pages-articles.xml.bz2'

def head(stream, n=10):
    """Convenience fnc: return the first `n` elements of the stream, as plain list."""
    return list(itertools.islice(stream, n))
    
from gensim.utils import smart_open, simple_preprocess
from gensim.corpora.wikicorpus import _extract_pages, filter_wiki
from gensim.parsing.preprocessing import STOPWORDS

def tokenize(text):
    return [token for token in simple_preprocess(text) if token not in STOPWORDS]

def iter_wiki(dump_file):
    """Yield each article from the Wikipedia dump, as a `(title, tokens)` 2-tuple."""
    ignore_namespaces = 'Wikipedia Category File Portal Template MediaWiki User Help Book Draft'.split()
    for title, text, pageid in _extract_pages(smart_open(dump_file)):
        text = filter_wiki(text)
        tokens = tokenize(text)
        if len(tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):          
            continue  # ignore short articles and various meta-articles
        yield title, tokens
        
# only use simplewiki in this tutorial (fewer documents)
# the full wiki dump is exactly the same format, but larger
stream = iter_wiki(corpus)
for title, tokens in itertools.islice(stream, 8):
    print title, tokens[:10]  # print the article title and its first ten tokens
    
id2word = {0: u'word', 2: u'profit', 300: u'another_word'}

doc_stream = (tokens for _, tokens in iter_wiki(corpus))

id2word_wiki = gensim.corpora.Dictionary(doc_stream)

id2word_wiki.filter_extremes(no_below=20, no_above=0.1)