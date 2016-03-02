# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 00:44:46 2016

@author: Sharon
"""
import os
import cPickle as pickle
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
os.chdir('C:\Users\Sharon\Documents\Statistics_CS\Stat141SL\Twitter')

cd 'C:\Users\Sharon\Documents\Statistics_CS\Stat141SL\Twitter'
cd 'C:\Users\Sharon\Documents\Statistics_CS\Stat141SL\Twitter'; python process_text.py
python process_text.py;  python lda_model.py
from process_text import load_documents, tokenize_documents, valid_word; import unicodecsv as csv; import nltk; from nltk.corpus import stopwords; from nltk.stem.snowball import EnglishStemmer; import re

documents = load_documents('data\\testorigtweets.csv')

with open('data\\andytweets_dtm.pkl') as pkl_file:
    corpus = pickle.load(pkl_file)

train, test = split_corpus(corpus, len(corpus), 0.9)

lda = LdaMulticore.load('data\\orig_5topics.lda')
lda2 = LdaMulticore.load('data\\andytweets_model.pkl')


id2word = corpora.Dictionary.load('data\\andytweets_dict.pkl')
id2word.save_as_text('data\\sorted_dict_andy.tsv', sort_by_word=False)

#process('libyan_tweets.csv')
#def main():
#    with open('params/process_text.yml', 'rb') as params_file:
#        params = yaml.load(params_file)
#    process(**params)
#
#if __name__ == '__main__':
#    main()