#!/usr/bin/env python

from gensim import corpora
import unicodecsv as csv
import cPickle as pickle
import string
import re
import time
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import EnglishStemmer
from nltk.stem.isri import ISRIStemmer

#import os

"""
Transformations:

* strip trailing whitespace
* remove punctuation/numbers
* convert to lowercase
* remove english stopwords
* remove suffixes (snowball stemmer)
* remove low frequency terms
"""
def load_documents(text_dump_path):
    with open(text_dump_path, 'rU') as f:
        reader = csv.DictReader(f, delimiter=',', dialect='excel')
        for row in reader:
            yield [row['row'], row['Text']] 

def tokenize_documents(documents):

    stop_words = stopwords.words('english') + stopwords.words('spanish') #common words to be filtered
    english = EnglishStemmer()
    arabic = ISRIStemmer()

    punctuation = { ord(char): None for char in string.punctuation}

    def valid_word(token, filtered=stop_words): 
        # Returns false for common words, links, and strange patterns
            if (token in filtered) or (token[0:4] == u'http') or\
            (token in string.punctuation):
                return False
            else:
                return True

    for doc in documents:

        row = doc[0]
        doc = doc[1]

        if doc is not None:

            # remove trailing whitespace
            doc = doc.strip()
            # remove twitter handles (words in doc starting with @)
            doc = re.sub(r"@\w+|\b@\w+", "", doc)
            # lowercase letters
            doc = doc.lower()
            # remove punctuation
            doc = doc.translate(punctuation)

            # tokenization: handles documents with arabic or foreign characters
            tokens = nltk.tokenize.wordpunct_tokenize(doc)

            cleaned_tokens = []
            for token in tokens:

                # for valid words, correct spellings of gaddafi and stem words
                if valid_word(token):
                
                    if token in [u'gadhafi', u'gadafi', u'ghadhafi', u'kadhafi', u'khadafi', u'kaddafi']:
                        token = u'gaddafi'
                    else:
                        token = arabic.stem(english.stem(token)) 

                    cleaned_tokens.append(token)    

            yield row
            yield cleaned_tokens
                 


def make_dtm(tokenized_docs, no_below=50, perc_above=0.95, keep_n=100000):
    #Filter out tokens that appear in
        #1. less than no_below documents (absolute number) or
        #2. more than perc_above documents (fraction of total corpus size, not absolute number).
        #3. after (1) and (2), keep only the first keep_n most frequent tokens (or keep all if None).
    id2word = corpora.Dictionary(tokenized_docs)
    id2word.filter_extremes(no_below, perc_above, keep_n)

    #convert document to bag-of-words representation: (word_id, frequency)
    dtm = [id2word.doc2bow(token_doc) for token_doc in tokenized_docs]

    return (id2word, dtm)

def main():
    start_time = time.time()
    documents = load_documents('data\\origtweets.csv')
    tokenized_docs = list(tokenize_documents(documents))
    rows = tokenized_docs[::2]
    tokenized_docs = tokenized_docs[1::2]
    id2word, dtm = make_dtm(tokenized_docs)

    id2word.save('data\\origtweets_dict.pkl')
    with open('data\\origtweets_dtm.pkl', 'wb') as pkl_file:
        pickle.dump(dtm, pkl_file)
    id2word.save_as_text('data\\sorted_dict.tsv', sort_by_word=False)

    f = open('data\\cleaned_origtweets.csv', 'wb')
    w = csv.writer(f, encoding='utf-8')
    w.writerow(['row', 'Text'])
    
    for i in xrange(len(rows)):
        doc = [rows[i], u' '.join(tokenized_docs[i])]
        w.writerow(doc)

    end_time = time.time()
    print ('running time: ' + str((end_time - start_time)/60) + ' minutes')

if __name__ == '__main__':
    main()
